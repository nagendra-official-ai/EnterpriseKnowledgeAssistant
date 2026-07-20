from evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from rag.retriever import RetrievalService


def main() -> None:
    retrieval_service = RetrievalService(
        result_count=5,
        persist_directory="vector_db/knowledge",
        collection_name="enterprise_knowledge",
        search_type="similarity",
    )

    evaluator = RetrievalEvaluator(retrieval_service=retrieval_service)

    summary = evaluator.evaluate_file("data/evaluation/retrieval_test_cases.json")

    print("\nRetrieval evaluation summary")
    print("=" * 70)
    print(f"Total cases: {summary.total_cases}")
    print(
        "Successful source hits:",
        summary.successful_source_hits,
    )
    print(f"Hit rate: {summary.hit_rate:.2%}")
    print(
        "Mean reciprocal rank:",
        f"{summary.mean_reciprocal_rank:.3f}",
    )
    print(
        "Average keyword coverage:",
        f"{summary.average_keyword_coverage:.2%}",
    )

    print("\nIndividual cases")

    for result in summary.case_results:
        print("\n" + "-" * 70)
        print(f"Case: {result.case_id}")
        print(f"Question: {result.question}")
        print(f"Source hit: {result.source_hit}")
        print(
            "First relevant rank:",
            result.first_relevant_rank,
        )
        print(
            "Reciprocal rank:",
            f"{result.reciprocal_rank:.3f}",
        )
        print(
            "Keyword coverage:",
            f"{result.keyword_coverage:.2%}",
        )
        print(
            "Retrieved sources:",
            result.retrieved_sources,
        )
        print(
            "Matched keywords:",
            result.matched_keywords,
        )

    assert summary.total_cases > 0
    assert 0.0 <= summary.hit_rate <= 1.0
    assert 0.0 <= summary.mean_reciprocal_rank <= 1.0
    assert 0.0 <= summary.average_keyword_coverage <= 1.0


if __name__ == "__main__":
    main()
