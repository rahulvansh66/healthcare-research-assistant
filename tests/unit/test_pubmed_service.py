import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import requests

from app.services.retrieval import pubmed_service


ESEARCH_JSON = {"esearchresult": {"idlist": ["111", "222"], "count": "2"}}

EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>First Study</ArticleTitle>
        <Abstract><AbstractText>Abstract one.</AbstractText></Abstract>
        <Journal>
          <Title>Journal A</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
        <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _mock_response(json_data=None, content=None, status_ok=True):
    resp = MagicMock(spec=requests.Response)
    resp.json.return_value = json_data
    resp.content = content
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("boom")
    return resp


@patch("app.services.retrieval.pubmed_service._throttle", lambda: None)
class TestSearchPubmed:
    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_returns_pmids_and_total(self, mock_get):
        mock_get.return_value = _mock_response(json_data=ESEARCH_JSON)

        result = pubmed_service.search_pubmed("widget therapy")

        assert result == {"pmids": ["111", "222"], "total_results": 2}

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_degrades_to_empty_on_request_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("down")

        result = pubmed_service.search_pubmed("widget therapy")

        assert result == {"pmids": [], "total_results": 0}

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_degrades_to_empty_on_malformed_json(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"unexpected": "shape"})

        result = pubmed_service.search_pubmed("widget therapy")

        assert result == {"pmids": [], "total_results": 0}

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_builds_query_with_date_range_and_publication_type(self, mock_get):
        mock_get.return_value = _mock_response(json_data=ESEARCH_JSON)

        pubmed_service.search_pubmed(
            "widget therapy",
            date_from="2020/01/01",
            date_to="2024/01/01",
            publication_type="Randomized Controlled Trial",
        )

        sent_term = mock_get.call_args.kwargs["params"]["term"]
        assert '("widget therapy")' in sent_term
        assert '("Randomized Controlled Trial"[Publication Type])' in sent_term
        assert '"2020/01/01"[Date - Publication] : "2024/01/01"[Date - Publication]' in sent_term


@patch("app.services.retrieval.pubmed_service._throttle", lambda: None)
class TestGetPubmedArticles:
    def test_empty_pmids_returns_empty_without_request(self):
        assert pubmed_service.get_pubmed_articles([]) == []

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_parses_article_fields(self, mock_get):
        mock_get.return_value = _mock_response(content=EFETCH_XML.encode())

        articles = pubmed_service.get_pubmed_articles(["111"])

        assert len(articles) == 1
        doc = articles[0]
        assert doc["pmid"] == "111"
        assert doc["title"] == "First Study"
        assert doc["abstract"] == "Abstract one."
        assert doc["journal"] == "Journal A"
        assert doc["year"] == "2021"
        assert doc["pub_types"] == ["Journal Article"]
        assert doc["authors"] == ["Smith J"]
        assert doc["has_full_text"] is False
        assert doc["full_text_markdown"] is None

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_degrades_to_empty_on_request_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("slow")

        assert pubmed_service.get_pubmed_articles(["111"]) == []

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_degrades_to_empty_on_malformed_xml(self, mock_get):
        mock_get.return_value = _mock_response(content=b"not xml at all <<<")

        assert pubmed_service.get_pubmed_articles(["111"]) == []

    @patch("app.services.retrieval.pubmed_service.requests.get")
    def test_skips_malformed_individual_records(self, mock_get):
        xml_with_no_pmid = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle><MedlineCitation><Article><ArticleTitle>No PMID</ArticleTitle></Article></MedlineCitation></PubmedArticle>
        </PubmedArticleSet>
        """
        mock_get.return_value = _mock_response(content=xml_with_no_pmid.encode())

        assert pubmed_service.get_pubmed_articles(["999"]) == []


def test_retry_then_success_returns_response():
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout("slow")
        resp = _mock_response(json_data=ESEARCH_JSON)
        return resp

    with patch("app.services.retrieval.pubmed_service._throttle", lambda: None), \
         patch("app.services.retrieval.pubmed_service.requests.get", side_effect=fake_get):
        result = pubmed_service.search_pubmed("widget therapy")

    assert calls["n"] == 2
    assert result == {"pmids": ["111", "222"], "total_results": 2}
