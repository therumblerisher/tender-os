# Tender OS Evaluation Report

Evaluated on 6899 cleaned public notices (6899 embedded with YTL `bge-m3`, 1,024 dims) across 15 Malaysian procurement portals.

## Untouched Holdout Retrieval (holdout_30, n=30)

| Mode | Overall MRR@10 | Overall R@10 | Overall R@50 | Exact MRR@10 | Malay MRR@10 | Cross-Lingual MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 0.5575 | 0.6333 | 0.6889 | 1.0000 | 0.9286 | 0.1483 |
| `vector` | 0.6634 | 0.7583 | 0.8361 | 0.1054 | 1.0000 | 0.8039 |
| `hybrid` | 0.7206 | 0.9250 | 1.0000 | 0.9375 | 0.9286 | 0.5078 |
| `hybrid_rerank` | 0.9500 | 1.0000 | 1.0000 | 0.8750 | 1.0000 | 0.9667 |

Missed in `hybrid_rerank` top-10 on `holdout_30`: 0

## Development Retrieval (dev_30, n=30)

| Mode | Overall MRR@10 | Overall R@10 | Overall R@50 | Exact MRR@10 | Malay MRR@10 | Cross-Lingual MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 0.4500 | 0.4905 | 0.6000 | 1.0000 | 0.7857 | 0.0000 |
| `vector` | 0.6239 | 0.6730 | 0.7171 | 0.3042 | 1.0000 | 0.6189 |
| `hybrid` | 0.6112 | 0.8516 | 0.9000 | 0.9375 | 0.7857 | 0.3557 |
| `hybrid_rerank` | 0.8319 | 0.8881 | 0.9000 | 0.9375 | 1.0000 | 0.6972 |

Missed in `hybrid_rerank` top-10 on `dev_30`: 3
- `d21` (english_cross_lingual): `mosque cleaning and janitorial maintenance services` (top hit: `['eperolehan:747e2d9dfcbf5ba9']`)
- `d22` (english_cross_lingual): `grass cutting and lawn mowing at government staff quarters` (top hit: `['eperolehan:f65e901030750ee8']`)
- `d23` (english_cross_lingual): `covered walkway link from pedestrian overhead bridge PKNS` (top hit: `['pkns:PKNS/PPS/400-4 (9)']`)

## Search Latency (5 interleaved trials x 6 dev queries, limit=10)

| Mode | Samples | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pg_fts` | 30 | 3.5 | 4.6 | 0.4 | 4.7 |
| `vector` | 30 | 101.4 | 191.9 | 82.6 | 194.2 |
| `hybrid` | 30 | 103.6 | 210.5 | 80.0 | 234.5 |
| `hybrid_rerank` | 30 | 225.4 | 1127.2 | 150.1 | 1521.6 |

## Postgres Query Plan (`EXPLAIN ANALYZE`, limit=50)

```text
Limit  (cost=1433.91..1434.04 rows=50 width=44) (actual time=21.610..21.615 rows=50 loops=1)
  Buffers: shared hit=47072
  ->  Sort  (cost=1433.91..1451.16 rows=6899 width=44) (actual time=21.608..21.611 rows=50 loops=1)
        Sort Key: ((embedding <=> '[-0.0038791304,0.055391908,-0.03314465,-0.01968317,-0.06542203,-0.030580556,-0.01649691,-0.0047628935,-0.031466678,0.021342287,-0.0016084487,0.06010531,-0.020286486,-0.0033465158,0.0206...
        Sort Method: top-N heapsort  Memory: 31kB
        Buffers: shared hit=47072
        ->  Seq Scan on tenders  (cost=0.00..1204.73 rows=6899 width=44) (actual time=0.019..21.052 rows=6899 loops=1)
              Filter: (embedding IS NOT NULL)
```

## End-to-End Agent Evaluation (dev vs untouched holdout)

| Split | Answerable Hits | OOS Abstentions | Pre-Gen OOS (0 chat calls) | Verified Cites | Unverified IDs | Date Mismatches | Fallbacks | p50 (ms) | Spend (MYR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched holdout (n=10) | 7/7 | 3/3 | 2/3 | 16 | 0 | 0 | 0 | 1189.3 | RM 0.015767 |
| Dev (n=10) | 5/7 | 3/3 | 3/3 | 13 | 0 | 0 | 0 | 912.6 | RM 0.014359 |
| Combined (n=20) | 12/14 | 6/6 | 5/6 | 29 | 0 | 0 | 0 | 1111.8 | RM 0.030126 |

Total recorded eval spend: RM 0.204205
