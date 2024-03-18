window.BENCHMARK_DATA = {
  "lastUpdate": 1710761939079,
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
      },
      {
        "commit": {
          "author": {
            "email": "137686240+ohad-starkware@users.noreply.github.com",
            "name": "Ohad",
            "username": "ohad-starkware"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "20a76bf117688c79648b48a4263c2a873d4b428d",
          "message": "integrated batch inverse to cpu twiddles (#469)",
          "timestamp": "2024-03-18T09:12:36+02:00",
          "tree_id": "7b2f69d21d950a600402b3b7f23db62e454d900e",
          "url": "https://github.com/starkware-libs/stwo/commit/20a76bf117688c79648b48a4263c2a873d4b428d"
        },
        "date": 1710746586679,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75916723,
            "range": "± 780171",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 175871098,
            "range": "± 20932678",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 255637267,
            "range": "± 2433564",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45782721,
            "range": "± 363262",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20741655,
            "range": "± 105174",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203065733,
            "range": "± 2921326",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46047244,
            "range": "± 1341615",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214997364,
            "range": "± 14213042",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105413047,
            "range": "± 2558013",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45575453,
            "range": "± 538001",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20929542,
            "range": "± 218632",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7717803,
            "range": "± 91653",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736374,
            "range": "± 25280",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4731997,
            "range": "± 11825",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 573966,
            "range": "± 8189",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 278214,
            "range": "± 6232",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 275987,
            "range": "± 17768",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 555347,
            "range": "± 5906",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 554354,
            "range": "± 3080",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1177368,
            "range": "± 6308",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1182428,
            "range": "± 9912",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2276640,
            "range": "± 43629",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2271558,
            "range": "± 16121",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4485824,
            "range": "± 42315",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4451973,
            "range": "± 40645",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "w.k@berkeley.edu",
            "name": "Weikeng Chen",
            "username": "weikengchen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b806150fccfc8f8a3e535ea85f2727e973699e4f",
          "message": "Reconcile the CM31 -> QM31 field extension polynomial with Plonky3 (#505)\n\n* replace the field extension polynomial\n\n* Merge branch 'dev' of github.com:weikengchen/stwo into dev",
          "timestamp": "2024-03-18T13:30:14+02:00",
          "tree_id": "467c36cdd91c53b267ec314cdbb6397cfcc597fb",
          "url": "https://github.com/starkware-libs/stwo/commit/b806150fccfc8f8a3e535ea85f2727e973699e4f"
        },
        "date": 1710761938322,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 68488524,
            "range": "± 1207604",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 217342655,
            "range": "± 1638441",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 237744569,
            "range": "± 3027817",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45608852,
            "range": "± 333977",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20734344,
            "range": "± 201916",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203946313,
            "range": "± 4597816",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46708805,
            "range": "± 1438701",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1252043939,
            "range": "± 72132513",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105775210,
            "range": "± 2327300",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45849361,
            "range": "± 1070982",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20832954,
            "range": "± 398968",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7752442,
            "range": "± 108481",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4773204,
            "range": "± 36303",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4755468,
            "range": "± 38388",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577814,
            "range": "± 8333",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 760,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 277128,
            "range": "± 3491",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 277680,
            "range": "± 8518",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 565770,
            "range": "± 8259",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 556442,
            "range": "± 5987",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1213898,
            "range": "± 16739",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1218086,
            "range": "± 22024",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2330805,
            "range": "± 31287",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2300225,
            "range": "± 19301",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4712387,
            "range": "± 56582",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4674772,
            "range": "± 44922",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}