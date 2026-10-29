from pangu_infer.pangu.accelerators.schedule_policy.normalized_scorer import TimeAndLengthScorer

if __name__ == "__main__":
    '''
    Assumptions:
    - Time: score is 0.5 at 10 seconds, approaching 1.0 at 80 seconds (k=1/10)
    - Length: score is 0.5 at 32K, approaching 1.0 at 256K (k=1/32K)
    '''
    scorer: TimeAndLengthScorer = TimeAndLengthScorer(
        time_median=10, time_weight=0.5,
        length_median=32*1024, length_weight=0.5
    )

    test_cases = [
        (5, 16*1024),
        (15, 64*1024),
        (80, 24*1024),
        (2, 256*1024),
    ]

    print("-" * 50)
    for t, l in test_cases:
        s = scorer.score(t, l)
        print(f"time:{t}, len:{l}, score:{s:.3f}")
