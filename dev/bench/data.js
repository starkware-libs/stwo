window.BENCHMARK_DATA = {
  "lastUpdate": 1710682115613,
  "repoUrl": "https://github.com/starkware-libs/stwo",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "43779613+spapinistarkware@users.noreply.github.com",
            "name": "Shahar Papini",
            "username": "spapinistarkware"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d666a1af3e6b9ae8811685ad674b0a444ca0286",
          "message": "Benchmarks in CI (#485)",
          "timestamp": "2024-03-17T11:23:31+02:00",
          "tree_id": "19178c353cee3455ea590fa208db7efe343f9593",
          "url": "https://github.com/starkware-libs/stwo/commit/3d666a1af3e6b9ae8811685ad674b0a444ca0286"
        },
        "date": 1710668903088,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76424077,
            "range": "± 1162730",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 237951936,
            "range": "± 7122647",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 269208410,
            "range": "± 3584567",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45624795,
            "range": "± 285894",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20864839,
            "range": "± 155153",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203845874,
            "range": "± 3669511",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46817391,
            "range": "± 1598864",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1223147113,
            "range": "± 10785811",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104957889,
            "range": "± 1613219",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45593919,
            "range": "± 192540",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20766662,
            "range": "± 147259",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7779837,
            "range": "± 126333",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4745710,
            "range": "± 15444",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4736223,
            "range": "± 12364",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 574444,
            "range": "± 4273",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 75",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 279551,
            "range": "± 10754",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 276880,
            "range": "± 2316",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 554716,
            "range": "± 4884",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 556225,
            "range": "± 6210",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1219511,
            "range": "± 37757",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1214438,
            "range": "± 14265",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2310702,
            "range": "± 28323",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2302181,
            "range": "± 10977",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4548452,
            "range": "± 63302",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4604044,
            "range": "± 144205",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43779613+spapinistarkware@users.noreply.github.com",
            "name": "Shahar Papini",
            "username": "spapinistarkware"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e9aa5fcddd571aaf9a2310f219d06b9ee050572d",
          "message": "Move commitment_scheme.rs (#499)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/499)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-17T15:19:58+02:00",
          "tree_id": "2d02c3ef7ea8b70f63cb340d54241ea8bbb66bd8",
          "url": "https://github.com/starkware-libs/stwo/commit/e9aa5fcddd571aaf9a2310f219d06b9ee050572d"
        },
        "date": 1710682114915,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76624592,
            "range": "± 800346",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 200614493,
            "range": "± 6571435",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 258236831,
            "range": "± 2296031",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45653506,
            "range": "± 201371",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20785434,
            "range": "± 449795",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204225239,
            "range": "± 3731813",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46147363,
            "range": "± 630491",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219714974,
            "range": "± 6812843",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104984205,
            "range": "± 4149553",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45818045,
            "range": "± 503623",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20809114,
            "range": "± 353014",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7710825,
            "range": "± 165426",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735897,
            "range": "± 11560",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733485,
            "range": "± 10911",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 574676,
            "range": "± 12479",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 276499,
            "range": "± 2496",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 276000,
            "range": "± 1729",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 553412,
            "range": "± 3674",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 551575,
            "range": "± 6052",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1202071,
            "range": "± 15311",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1214440,
            "range": "± 20373",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2279551,
            "range": "± 16496",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2281839,
            "range": "± 11267",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4471145,
            "range": "± 44838",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4490763,
            "range": "± 78054",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}