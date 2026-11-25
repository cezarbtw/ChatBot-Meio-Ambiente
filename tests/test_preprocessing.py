"""Testes para o módulo compartilhado de pré-processamento textual."""

from src.nlp.preprocessing import clean_text, combine_title_and_text


def test_clean_text_removes_urls_html_and_normalizes_whitespace():
    raw = "  <p>Alerta em SP:</p> veja mais em https://g1.globo.com/meio-ambiente/ e www.exemplo.org\n\tagora!  "
    assert clean_text(raw) == "alerta em sp: veja mais em e agora!"


def test_clean_text_preserves_portuguese_accents_and_lowercases():
    raw = "Amazônia registra REDUÇÃO de incêndios e poluição."
    assert clean_text(raw) == "amazônia registra redução de incêndios e poluição."


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text(float("nan")) == ""


def test_combine_title_and_text():
    assert (
        combine_title_and_text("  Novo Parque ", " Protege <b>floresta</b>  ")
        == "novo parque. protege floresta"
    )
    assert combine_title_and_text("Somente título", "") == "somente título"
    assert combine_title_and_text("", "Somente texto") == "somente texto"
