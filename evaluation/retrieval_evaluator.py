import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag.retriever import RetrievalService
from utils.logger import setup_logger


@dataclass
class RetrievalTestCase:
    """
    Defines one expected retrieval behavior.
    """

    case_id: str
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]


@dataclass
class RetrievalCaseResult:
    """
    Contains evaluation results for one test case.
    """

    case_id: str
    question: str
    retrieved_count: int
    source_hit: bool
    first_relevant_rank: int | None
    reciprocal_rank: float
    keyword_coverage: float
    retrieved_sources: List[str]
    matched_keywords: List[str]


@dataclass
class RetrievalEvaluationSummary:
    """
    Aggregated metrics across all retrieval test cases.
    """

    total_cases: int
    successful_source_hits: int
    hit_rate: float
    mean_reciprocal_rank: float
    average_keyword_coverage: float
    case_results: List[RetrievalCaseResult]


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using expected source files
    and expected content keywords.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
    ) -> None:
        self.logger = setup_logger()
        self.retrieval_service = retrieval_service

    def evaluate_file(
        self,
        test_data_path: str,
    ) -> RetrievalEvaluationSummary:
        """
        Load test cases from JSON and evaluate them.
        """
        test_cases = self._load_test_cases(test_data_path)

        return self.evaluate(test_cases)

    def evaluate(
        self,
        test_cases: List[RetrievalTestCase],
    ) -> RetrievalEvaluationSummary:
        """
        Evaluate multiple retrieval test cases.
        """
        if not test_cases:
            raise ValueError("At least one retrieval test case is required.")

        case_results: List[RetrievalCaseResult] = []

        for test_case in test_cases:
            result = self._evaluate_case(test_case)

            case_results.append(result)

        source_hits = sum(1 for result in case_results if result.source_hit)

        hit_rate = source_hits / len(case_results)

        mean_reciprocal_rank = sum(
            result.reciprocal_rank for result in case_results
        ) / len(case_results)

        average_keyword_coverage = sum(
            result.keyword_coverage for result in case_results
        ) / len(case_results)

        summary = RetrievalEvaluationSummary(
            total_cases=len(case_results),
            successful_source_hits=source_hits,
            hit_rate=hit_rate,
            mean_reciprocal_rank=(mean_reciprocal_rank),
            average_keyword_coverage=(average_keyword_coverage),
            case_results=case_results,
        )

        self.logger.info(
            "Retrieval evaluation completed. "
            "Cases: %d, Hit rate: %.2f, "
            "MRR: %.2f, Keyword coverage: %.2f",
            summary.total_cases,
            summary.hit_rate,
            summary.mean_reciprocal_rank,
            summary.average_keyword_coverage,
        )

        return summary

    def _evaluate_case(
        self,
        test_case: RetrievalTestCase,
    ) -> RetrievalCaseResult:
        """
        Evaluate one question against its expected evidence.
        """
        self._validate_test_case(test_case)

        documents = self.retrieval_service.retrieve(test_case.question)

        retrieved_sources = [self._extract_filename(document) for document in documents]

        normalized_expected_sources = {
            source.lower() for source in test_case.expected_sources
        }

        first_relevant_rank = self._find_first_relevant_rank(
            retrieved_sources=(retrieved_sources),
            expected_sources=(normalized_expected_sources),
        )

        source_hit = first_relevant_rank is not None

        reciprocal_rank = (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )

        combined_content = " ".join(
            document.page_content.lower() for document in documents
        )

        matched_keywords = [
            keyword
            for keyword in test_case.expected_keywords
            if keyword.lower() in combined_content
        ]

        keyword_coverage = (
            len(matched_keywords) / len(test_case.expected_keywords)
            if test_case.expected_keywords
            else 1.0
        )

        result = RetrievalCaseResult(
            case_id=test_case.case_id,
            question=test_case.question,
            retrieved_count=len(documents),
            source_hit=source_hit,
            first_relevant_rank=(first_relevant_rank),
            reciprocal_rank=reciprocal_rank,
            keyword_coverage=keyword_coverage,
            retrieved_sources=retrieved_sources,
            matched_keywords=matched_keywords,
        )

        self.logger.info(
            "Evaluated retrieval case '%s'. "
            "Source hit: %s, rank: %s, "
            "keyword coverage: %.2f",
            result.case_id,
            result.source_hit,
            result.first_relevant_rank,
            result.keyword_coverage,
        )

        return result

    @staticmethod
    def _load_test_cases(
        test_data_path: str,
    ) -> List[RetrievalTestCase]:
        """
        Read retrieval test cases from a JSON file.
        """
        path = Path(test_data_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: " f"{test_data_path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_test_cases = json.load(file)

        if not isinstance(raw_test_cases, list):
            raise ValueError("Evaluation JSON must contain a list.")

        return [
            RetrievalTestCase(
                case_id=item["case_id"],
                question=item["question"],
                expected_sources=item.get(
                    "expected_sources",
                    [],
                ),
                expected_keywords=item.get(
                    "expected_keywords",
                    [],
                ),
            )
            for item in raw_test_cases
        ]

    @staticmethod
    def _extract_filename(
        document: Document,
    ) -> str:
        """
        Extract only the source filename from metadata.
        """
        source = str(
            document.metadata.get(
                "source",
                "Unknown",
            )
        )

        return Path(source).name

    @staticmethod
    def _find_first_relevant_rank(
        retrieved_sources: List[str],
        expected_sources: set[str],
    ) -> int | None:
        """
        Return the 1-based rank of the first expected source.
        """
        for rank, source in enumerate(
            retrieved_sources,
            start=1,
        ):
            if source.lower() in expected_sources:
                return rank

        return None

    @staticmethod
    def _validate_test_case(
        test_case: RetrievalTestCase,
    ) -> None:
        if not test_case.case_id.strip():
            raise ValueError("Test case ID cannot be empty.")

        if not test_case.question.strip():
            raise ValueError("Test question cannot be empty.")

        if not test_case.expected_sources:
            raise ValueError(
                f"Test case '{test_case.case_id}' "
                "requires at least one expected source."
            )
