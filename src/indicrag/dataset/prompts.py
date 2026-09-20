"""Prompts and grammars for bootstrapping question-answer candidates.

Four instructions carry almost all the weight here, and each answers a specific
observed failure of small instruction-tuned models on this task:

1. **Ask about a fact stated in the passage.** Without it the model invents
   plausible scheme details that are not in the text, which produces a gold
   answer no retrieval system could ever support.
2. **Do not reuse the passage's distinctive wording.** This is the leakage
   control from `docs/PRD.md` §6.6 applied at generation time rather than only
   measured afterwards. A question that quotes its own passage hands BM25 the
   answer and makes the dense-vs-lexical comparison meaningless.
3. **Copy the answer verbatim.** Numbers, dates and scheme names must match the
   document exactly, because Exact Match and token-F1 are scored against them.
   Paraphrased gold answers silently depress every QA metric.
4. **Answer in the passage's language.** A Hindi passage must yield a Hindi
   question, or the language-pair matrix in §6.2 cannot be filled.

Output is constrained by a GBNF grammar rather than merely requested as JSON.
A 1.5B model asked politely for JSON produces malformed output often enough that
the repair pass becomes its own source of bias -- dropping unparseable generations
silently biases the set towards the easy passages.
"""

from __future__ import annotations

#: GBNF grammar: the model cannot emit anything but this object shape.
QA_GRAMMAR = r"""
root   ::= "{" ws "\"question\"" ws ":" ws string ws "," ws
                "\"answer\"" ws ":" ws string ws "," ws
                "\"kind\"" ws ":" ws kind ws "}"
kind   ::= "\"fact\"" | "\"number\"" | "\"date\"" | "\"eligibility\"" | "\"process\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
"""

_SHARED_RULES = """Rules:
- Ask about a fact that is stated in the passage. Never invent details.
- Do not reuse the passage's distinctive wording; rephrase the question.
- The answer must be copied from the passage exactly, including numbers and names.
- Keep the question under 20 words and the answer under 25 words.
- If the passage states no specific fact worth asking about, answer with
  {"question": "", "answer": "", "kind": "fact"}."""

PROMPT_EN = """You write exam questions from official documents about Indian government schemes.

Passage ({scheme}, section: {section}):
\"\"\"
{passage}
\"\"\"

Write ONE question in English that this passage answers, and the answer.

{rules}

Reply with JSON only."""

PROMPT_HI = """आप भारत सरकार की योजनाओं के दस्तावेज़ों से प्रश्न बनाते हैं।

अनुच्छेद ({scheme}, अनुभाग: {section}):
\"\"\"
{passage}
\"\"\"

इस अनुच्छेद से हिन्दी में एक प्रश्न और उसका उत्तर लिखिए।

नियम:
- प्रश्न उसी तथ्य पर हो जो अनुच्छेद में लिखा है। कुछ भी अपनी ओर से न जोड़ें।
- अनुच्छेद के शब्द ज्यों के त्यों न दोहराएँ; प्रश्न को अपने शब्दों में लिखें।
- उत्तर अनुच्छेद से हूबहू लिया जाए, संख्याएँ और नाम बिल्कुल वैसे ही।
- प्रश्न 20 शब्दों से कम और उत्तर 25 शब्दों से कम रखें। संक्षिप्त लिखें।
- प्रश्न और उत्तर हिन्दी (देवनागरी) में ही लिखें।

JSON की कुंजियाँ (keys) अंग्रेज़ी में ही रहें, अनुवाद न करें:
{{"question": "...", "answer": "...", "kind": "fact"}}

केवल यही JSON लौटाएँ, और कुछ नहीं।"""

#: Hinglish is produced by transliterating a Hindi question rather than by asking
#: the model for code-mix directly. Small models write a stilted, over-formal
#: register when asked to "write in Hinglish" that looks nothing like how people
#: actually type, and docs/PRD.md §6.5 step 2 requires these be hand-written or
#: hand-corrected. This gives the annotator a realistic starting point to correct.
PROMPT_HINGLISH = """Rewrite this Hindi question the way an Indian user would actually type it on a phone: Roman script, mixing English words where people normally do.

Hindi question: {question}

Keep the meaning identical. Do not translate fully to English -- keep Hindi grammar words
(kya, hai, ke liye, kitna, kaise, kab) in Roman script, and keep English terms that
people normally use in English (scholarship, eligibility, income, account, online).

Reply with JSON only, as {{"question": "...", "answer": "", "kind": "fact"}}."""


#: A Hindi prompt makes a model translate the JSON KEYS as well as the content:
#: Qwen2.5-3B returned {"प्रश्न": ..., "उत्तर": ...} -- well-formed JSON that
#: closed properly, and completely unparseable by a reader looking for
#: "question". The keys are therefore pinned in English inside the Hindi prompt,
#: with the literal object shape shown. Diagnosed by reading the raw output; the
#: parse-failure counter alone would have said "Hindi fails" and pointed at the
#: model rather than at the prompt.
#:
#: Cross-lingual items are made by generating the question in the *passage's*
#: language and then translating the question alone, leaving the gold passage
#: where it is. That is what makes the item genuinely cross-lingual: the evidence
#: exists only in the other language, so a retriever must cross the boundary
#: rather than find a same-language paraphrase. Translating the passage instead
#: would quietly turn a cross-lingual probe into a monolingual one.
PROMPT_TRANSLATE_TO_EN = """Translate this Hindi question into natural English. Keep scheme names, numbers and dates exactly as they are.

Hindi question: {question}

Reply with JSON only, as {{"question": "...", "answer": "", "kind": "fact"}}."""

PROMPT_TRANSLATE_TO_HI = """इस अंग्रेज़ी प्रश्न का हिन्दी में अनुवाद कीजिए। योजना के नाम, संख्याएँ और तिथियाँ ज्यों की त्यों रखिए।

अंग्रेज़ी प्रश्न: {question}

केवल JSON में उत्तर दें: {{"question": "...", "answer": "", "kind": "fact"}}"""


def build_translate_prompt(question: str, *, to: str) -> str:
    template = PROMPT_TRANSLATE_TO_EN if to == "en" else PROMPT_TRANSLATE_TO_HI
    return template.format(question=question)


def build_prompt(*, lang: str, scheme: str, section: str, passage: str) -> str:
    template = PROMPT_HI if lang == "hi" else PROMPT_EN
    if lang == "hi":
        return template.format(scheme=scheme, section=section or "मुख्य", passage=passage)
    return template.format(
        scheme=scheme, section=section or "main", passage=passage, rules=_SHARED_RULES
    )


def build_hinglish_prompt(question: str) -> str:
    return PROMPT_HINGLISH.format(question=question)


#: Unanswerable items are NOT generated by the model. docs/PRD.md §6.3 requires
#: the near-miss class be authored while looking at the passage it is designed to
#: nearly match, and a model asked for an unanswerable question reliably produces
#: the trivial out-of-scope kind instead -- which proves nothing, because any
#: threshold rejects it. These stems are offered to the annotator as scaffolding
#: for the four classes; the annotator supplies the scheme-specific detail.
UNANSWERABLE_SCAFFOLDS = {
    "out-of-scope": [
        "What is the current home loan interest rate?",
        "Who won the last general election?",
        "Train ka ticket kaise book karte hain?",
        "मोबाइल फोन पर जीएसटी कितना है?",
    ],
    "near-miss": [
        "What is the exact application deadline for {scheme} in 2026?",
        "How many applicants were rejected under {scheme} last year?",
        "{scheme} ka helpline number kya hai?",
        "{scheme} के लिए आवेदन शुल्क कितना है?",
    ],
    "false-premise": [
        "How much laptop allowance does {scheme} provide?",
        "What is the foreign travel grant under {scheme}?",
        "{scheme} ke tahat free hostel kitne din milta hai?",
        "{scheme} में विदेश यात्रा भत्ता कितना है?",
    ],
    "under-specified": [
        "What is the income limit?",
        "How much money will I get?",
        "Eligibility kya hai?",
        "आवेदन की अंतिम तिथि क्या है?",
    ],
}
