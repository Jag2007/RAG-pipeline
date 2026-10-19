from conftest import SAMPLE_PDF
from fakes import FakeEmbedder, FakeLLM

from rag.pipeline import RAGPipeline
from rag.prompt import NOT_FOUND


def make_pipeline(reply, tmp_path=None, min_similarity=0.1):
    pipeline = RAGPipeline(embedder=FakeEmbedder(), llm=FakeLLM(reply),
                           data_dir=tmp_path, min_similarity=min_similarity)
    pipeline.add_pdf(SAMPLE_PDF)
    return pipeline


def test_answer_is_grounded_with_citation():
    pipeline = make_pipeline("Keep the tyres between 40 and 55 psi [1].")
    result = pipeline.ask("What tyre pressure should I use?")
    assert result["grounded"]
    assert result["cited"]
    assert result["sources"][result["cited"][0] - 1]["page"] == 5


def test_prompt_contains_numbered_sources():
    pipeline = make_pipeline("anything")
    pipeline.ask("What tyre pressure should I use?")
    prompt = pipeline.llm.last_messages[-1]["content"]
    assert "[1] (sample_manual.pdf, page" in prompt
    assert "Question: What tyre pressure should I use?" in prompt


def test_unrelated_question_skips_the_llm():
    pipeline = make_pipeline("should not be used", min_similarity=0.99)
    result = pipeline.ask("What is the capital of France?")
    assert result["answer"] == NOT_FOUND
    assert pipeline.llm.last_messages is None


def test_model_refusal_is_passed_through():
    pipeline = make_pipeline(NOT_FOUND)
    result = pipeline.ask("What is the price of the bike?")
    assert result["answer"] == NOT_FOUND and not result["grounded"]


def test_made_up_answer_is_not_grounded():
    pipeline = make_pipeline("The bike costs 90,000 rupees and comes in purple.")
    assert not pipeline.ask("How much is it?")["grounded"]


def test_index_is_saved_and_reloaded(tmp_path):
    make_pipeline("x", tmp_path=tmp_path)
    reloaded = RAGPipeline(embedder=FakeEmbedder(), llm=FakeLLM("x"), data_dir=tmp_path)
    assert reloaded.documents() == [{"source": "sample_manual.pdf", "chunks": 12}]


def test_keyword_only_mode_without_embeddings():
    # how the app runs on Render with only a Groq key
    pipeline = RAGPipeline(llm=FakeLLM("Keep the tyres between 40 and 55 psi [1]."),
                           data_dir=None, provider="groq")
    pipeline.add_pdf(SAMPLE_PDF)
    result = pipeline.ask("What tyre pressure should I use?")
    assert result["grounded"]
    assert result["sources"][0]["page"] == 5
    assert result["sources"][0]["similarity"] is None


def test_keyword_only_mode_uses_rewritten_query():
    # "sop" is not in the manual, but the rewrite adds words that are
    llm = FakeLLM("Keep the tyres between 40 and 55 psi [1].", keywords="tyre pressure psi")
    pipeline = RAGPipeline(llm=llm, data_dir=None, provider="groq")
    pipeline.add_pdf(SAMPLE_PDF)
    result = pipeline.ask("what about sop?")
    assert result["sources"][0]["page"] == 5


def test_keyword_only_mode_falls_back_to_document_opening():
    llm = FakeLLM(NOT_FOUND, keywords="")
    pipeline = RAGPipeline(llm=llm, data_dir=None, provider="groq")
    pipeline.add_pdf(SAMPLE_PDF)
    result = pipeline.ask("capital of France?")
    assert result["answer"] == NOT_FOUND
    # the model was shown the start of the manual and refused
    assert "[1] (sample_manual.pdf, page 1)" in llm.last_messages[-1]["content"]


def test_reupload_replaces_instead_of_duplicating():
    pipeline = make_pipeline("x")
    pipeline.add_pdf(SAMPLE_PDF)
    assert pipeline.documents()[0]["chunks"] == 12
