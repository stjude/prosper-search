import argparse
from pathlib import Path
from typing import Dict, Hashable, Optional, Set

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)


def get_top_k_similar_slides(
    similarity_df: pd.DataFrame,
    top_k: int = 5,
    exclude_self: bool = True,
) -> Dict[Hashable, Set[Hashable]]:
    """
    Find the top-k most similar slides for each slide in a similarity matrix.

    Parameters
    ----------
    similarity_df
        n x n dataframe where rows and columns are slide IDs and values are
        similarity scores (lower means more similar).
    top_k
        Number of most similar slides to find.
    exclude_self
        If True, the slide itself will not appear in its own neighbor set.

    Returns
    -------
    dict
        Dictionary mapping each slide ID to a set of its top-k neighbor slide IDs.
    """
    top_k_neighbors: Dict[Hashable, Set[Hashable]] = {}

    for slide_id in similarity_df.index:
        row = similarity_df.loc[slide_id].sort_values()

        if exclude_self and slide_id in row.index:
            row = row.drop(slide_id)

        top_similar = row.index[:top_k].tolist()
        top_k_neighbors[slide_id] = set(top_similar)

    return top_k_neighbors


def solve_ilp(
    top_k_neighbors: Dict[Hashable, Set[Hashable]],
    max_slides: Optional[int] = None,
    time_limit: Optional[int] = None,
    solver_verbose: bool = True,
) -> Set[Hashable]:
    """
    Solve the minimum covering slides problem using Integer Linear Programming.

    Each slide must either be selected itself or be "covered" by a selected neighbor.

    Parameters
    ----------
    top_k_neighbors
        Dictionary mapping each slide ID to a set of its neighbor slide IDs.
        If A is in the neighbor set of B, then B can "cover" A.
    max_slides
        Optional upper bound on the number of selected slides.
        If None, no explicit limit is imposed.
    time_limit
        Optional solver time limit in seconds.
    solver_verbose
        If True, the solver prints log messages.

    Returns
    -------
    set
        Set of selected slide IDs.
    """
    print("Building ILP model...")
    slide_ids = list(top_k_neighbors.keys())

    # Build reverse lookup: for each slide, who can cover it
    covered_by: Dict[Hashable, Set[Hashable]] = {slide: set() for slide in slide_ids}
    for slide, neighbors in top_k_neighbors.items():
        for neighbor in neighbors:
            covered_by[neighbor].add(slide)

    # ILP model
    model = LpProblem("Minimum_Covering_Slides", LpMinimize)

    # Binary decision variables: x[slide] = 1 if slide is selected
    x = {slide: LpVariable(f"x_{slide}", cat=LpBinary) for slide in slide_ids}

    # Objective: minimize number of selected slides
    total_selected = lpSum(x[slide] for slide in slide_ids)
    model += total_selected

    # Optional limit on total number of slides
    if max_slides is not None:
        print(f"Adding max slides constraint: {max_slides}")
        model += total_selected <= max_slides

    # Coverage constraints: each slide must be selected or covered by a selected neighbor
    for slide in slide_ids:
        model += x[slide] + lpSum(x[neighbor] for neighbor in covered_by[slide]) >= 1

    # Solve model
    solver = PULP_CBC_CMD(msg=solver_verbose, timeLimit=time_limit)
    model.solve(solver)

    # Extract selected slides
    selected_slides = {slide for slide in slide_ids if x[slide].varValue == 1}

    return selected_slides


def main(
    sim_matrix_path: str,
    top_k: int = 5,
    max_slides: Optional[int] = None,
    time_limit: Optional[int] = None,
    output_path: Optional[str] = None,
    iterations: int = 1,
) -> None:

    sim_matrix_path = Path(sim_matrix_path)

    print(f"Loading similarity matrix from: {sim_matrix_path}")
    sim_matrix = pd.read_pickle(sim_matrix_path)

    current_matrix = sim_matrix.copy()
    prev_count = None

    for it in range(1, iterations + 1):
        print(f"\nIteration {it} of {iterations}")
        print(f"Matrix size: {current_matrix.shape}")

        print(f"Computing top-{top_k} neighbors")
        top_k_neighbors = get_top_k_similar_slides(current_matrix, top_k=top_k)

        print("Solving ILP")
        selected_slides = solve_ilp(
            top_k_neighbors=top_k_neighbors,
            max_slides=max_slides,
            time_limit=time_limit,
            solver_verbose=True,
        )

        print(f"Selected {len(selected_slides)} slides")

        # Save output for this iteration
        if output_path is None:
            suffix = "unbounded" if max_slides is None else f"max{max_slides}"
            out = (
                sim_matrix_path.parent
                / f"selected_slides_ilp_topk_{top_k}_{suffix}.txt"
            )
        else:
            out = Path(output_path).with_name(
                Path(output_path).stem + f".txt"
            )

        print(f"Saving iteration {it} results to {out}")
        with out.open("w") as f:
            for slide in selected_slides:
                f.write(f"{slide}\n")

        # Check if we should continue iterating
        if prev_count is not None and len(selected_slides) >= prev_count:
            print("Stopping early because slide count did not decrease.")
            break

        prev_count = len(selected_slides)

        # Reduce similarity matrix for next iteration
        selected_list = list(selected_slides)
        current_matrix = current_matrix.loc[selected_list, selected_list]

    print("\nFinished all iterations.")



if __name__ == "__main__":
    print("Starting slide selection using ILP...")
    parser = argparse.ArgumentParser(
        description="Compute a minimal covering set of slides using ILP."
    )
    parser.add_argument(
        "--sim-matrix-path",
        type=str,
        required=True,
        help="Path to the pickled similarity matrix DataFrame.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbors used for coverage (default: 5).",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="Optional upper bound on the number of selected slides.",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default=None,
        help="Optional CBC time limit in seconds.",
    )

    parser.add_argument(
    "--iter",
    type=int,
    default=1,
    help="Number of ILP refinement iterations to run.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save the selected slide IDs.",
    )

    args = parser.parse_args()
    main(
        sim_matrix_path=args.sim_matrix_path,
        top_k=args.top_k,
        max_slides=args.max_slides,
        time_limit=args.time_limit,
        output_path=args.output_path,
        iterations=args.iter,
    )

