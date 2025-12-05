from functools import lru_cache
from typing import List, Optional, Dict, Any

from dupes.data.pipeline import load_clean_dataset
from dupes.recommender import Recommendation, SimilarityRecommender


@lru_cache(maxsize=1)
def _get_recommender() -> SimilarityRecommender:
    df = load_clean_dataset()
    return SimilarityRecommender(df)


def predict_shampoo(
    shampoo: Optional[str] = None,
    description: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return a list of similar or cheaper products based on ingredients +
    description similarity.
    """
    recommender = _get_recommender()
    recs: List[Recommendation] = recommender.recommend(
        shampoo=shampoo,
        description=description,
        top_k=top_k,
    )
    return [
        {
            "product_id": rec.product_id,
            "product_name": rec.product_name,
            "price_eur": rec.price_eur,
            "similarity": rec.similarity,
        }
        for rec in recs
    ]
