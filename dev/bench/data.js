window.BENCHMARK_DATA = {
  "lastUpdate": 1715786911609,
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
          "id": "91ea3853376a696e608b6cc733d52f1c8ce2e929",
          "message": "Air with dyn component (#487)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/487)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-19T17:11:17+02:00",
          "tree_id": "f68c468a714be060f38131657e5c9bc9f85b8132",
          "url": "https://github.com/starkware-libs/stwo/commit/91ea3853376a696e608b6cc733d52f1c8ce2e929"
        },
        "date": 1710861872888,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 79537914,
            "range": "± 1190678",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 224459416,
            "range": "± 5393146",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 273738613,
            "range": "± 2529890",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45689220,
            "range": "± 661422",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20988313,
            "range": "± 761578",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204003132,
            "range": "± 1704291",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46174653,
            "range": "± 350135",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213710191,
            "range": "± 12534764",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105165260,
            "range": "± 1245016",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45865955,
            "range": "± 798337",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20783993,
            "range": "± 481839",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7748218,
            "range": "± 209723",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734566,
            "range": "± 8237",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729533,
            "range": "± 16402",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576778,
            "range": "± 14778",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 275571,
            "range": "± 1686",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 277037,
            "range": "± 2016",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 554333,
            "range": "± 14792",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 557390,
            "range": "± 9663",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1228216,
            "range": "± 14758",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1228678,
            "range": "± 8692",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2356600,
            "range": "± 28405",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2335334,
            "range": "± 53128",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4602493,
            "range": "± 84548",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4576418,
            "range": "± 52723",
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
          "id": "4d3c04d2fb079ebc54a8287e6652e3e4d02f74a7",
          "message": "CommitmentScheme utils (#500)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/500)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-19T17:17:37+02:00",
          "tree_id": "d86daa0f4495b8bb99a7ba2318f3c67de9d6fb82",
          "url": "https://github.com/starkware-libs/stwo/commit/4d3c04d2fb079ebc54a8287e6652e3e4d02f74a7"
        },
        "date": 1710862049540,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77665851,
            "range": "± 1043353",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 222442084,
            "range": "± 3748000",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 266606145,
            "range": "± 3462293",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45827152,
            "range": "± 1750054",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20810617,
            "range": "± 113305",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203547958,
            "range": "± 2127434",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46147278,
            "range": "± 978920",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219335985,
            "range": "± 21941620",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105142629,
            "range": "± 1432001",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45918216,
            "range": "± 646926",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20826937,
            "range": "± 671329",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7711212,
            "range": "± 44570",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4744622,
            "range": "± 10433",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738979,
            "range": "± 21743",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 579060,
            "range": "± 13221",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 277538,
            "range": "± 4274",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 277756,
            "range": "± 4884",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 558962,
            "range": "± 5019",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 556944,
            "range": "± 11008",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1220481,
            "range": "± 47367",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1230231,
            "range": "± 24524",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2309608,
            "range": "± 59513",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2356621,
            "range": "± 24435",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4557567,
            "range": "± 63195",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4578686,
            "range": "± 44858",
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
          "id": "828bdf38f8ffa4ef984366d928b81fb8e1a9025a",
          "message": "Field debug implementations (#501)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/501)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-19T17:21:37+02:00",
          "tree_id": "baccb78d61c2764d5534fc10a1ff8d8a27c907fd",
          "url": "https://github.com/starkware-libs/stwo/commit/828bdf38f8ffa4ef984366d928b81fb8e1a9025a"
        },
        "date": 1710862230063,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 84029148,
            "range": "± 1521528",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 247145113,
            "range": "± 4625545",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 282925708,
            "range": "± 1692273",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45856183,
            "range": "± 541513",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21036451,
            "range": "± 624354",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204289373,
            "range": "± 3925056",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46451077,
            "range": "± 752562",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216697193,
            "range": "± 17588747",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105645768,
            "range": "± 4469507",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45592353,
            "range": "± 715378",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20845715,
            "range": "± 620977",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7740877,
            "range": "± 168530",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734175,
            "range": "± 10148",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4742616,
            "range": "± 11816",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578214,
            "range": "± 10003",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 628,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 276774,
            "range": "± 1888",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 278754,
            "range": "± 7742",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 554453,
            "range": "± 3969",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 552689,
            "range": "± 6916",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1173164,
            "range": "± 24637",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1178308,
            "range": "± 4591",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2322894,
            "range": "± 33880",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2325814,
            "range": "± 26905",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4748816,
            "range": "± 44407",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4814802,
            "range": "± 170594",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b89a453c0764816218dff0208944666f025324c2",
          "message": "Implement 'Sum' and 'Product' on field types (#497)",
          "timestamp": "2024-03-20T09:22:47+02:00",
          "tree_id": "7ce13cdb1511189d634f614d0f07eeea186bc231",
          "url": "https://github.com/starkware-libs/stwo/commit/b89a453c0764816218dff0208944666f025324c2"
        },
        "date": 1710920001215,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73230832,
            "range": "± 1145654",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 206995720,
            "range": "± 3445186",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 256225348,
            "range": "± 2790932",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45662957,
            "range": "± 740152",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20873506,
            "range": "± 177680",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203995004,
            "range": "± 1158050",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46015959,
            "range": "± 928696",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1226784934,
            "range": "± 10789966",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104736923,
            "range": "± 1032180",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45813186,
            "range": "± 528018",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20844973,
            "range": "± 2487967",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7755873,
            "range": "± 53746",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4744196,
            "range": "± 14364",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4734090,
            "range": "± 14590",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578774,
            "range": "± 15457",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 277974,
            "range": "± 2051",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 275088,
            "range": "± 2909",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 551075,
            "range": "± 9156",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 557391,
            "range": "± 12001",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1201928,
            "range": "± 13177",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1204957,
            "range": "± 14994",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2294664,
            "range": "± 16793",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2281664,
            "range": "± 26028",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4499859,
            "range": "± 87769",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4463227,
            "range": "± 26752",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2b9653e0b60379346fc1d427f8445828bcee4e89",
          "message": "Fix dev. (#512)",
          "timestamp": "2024-03-20T11:22:56+02:00",
          "tree_id": "5e78beceebee71935709cc8b49c7e19704db1b56",
          "url": "https://github.com/starkware-libs/stwo/commit/2b9653e0b60379346fc1d427f8445828bcee4e89"
        },
        "date": 1710927155744,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75824032,
            "range": "± 1226390",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 157155877,
            "range": "± 6808083",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 250091478,
            "range": "± 2243771",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45649764,
            "range": "± 406601",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20715191,
            "range": "± 324458",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203306021,
            "range": "± 4820464",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46181673,
            "range": "± 553642",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216853811,
            "range": "± 18820345",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105268130,
            "range": "± 2040862",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45600346,
            "range": "± 297518",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20713327,
            "range": "± 502305",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7762443,
            "range": "± 53830",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734043,
            "range": "± 9056",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730084,
            "range": "± 13279",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577522,
            "range": "± 9811",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 276451,
            "range": "± 4738",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 275988,
            "range": "± 3853",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 553622,
            "range": "± 4282",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 553439,
            "range": "± 5688",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1201244,
            "range": "± 9275",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1199634,
            "range": "± 28816",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2285009,
            "range": "± 19515",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2294639,
            "range": "± 22129",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4450186,
            "range": "± 73873",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4437161,
            "range": "± 53556",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3bc4c79140c72a1beb244c919076868e2135abba",
          "message": "Create MultiFibonacci. (#509)",
          "timestamp": "2024-03-20T14:13:21+02:00",
          "tree_id": "bee14e384df7c4600d7f1aac224d749cea135827",
          "url": "https://github.com/starkware-libs/stwo/commit/3bc4c79140c72a1beb244c919076868e2135abba"
        },
        "date": 1710937436314,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78399087,
            "range": "± 3078205",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 228070697,
            "range": "± 11779614",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 274309391,
            "range": "± 9081892",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45653334,
            "range": "± 931978",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20720752,
            "range": "± 197721",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204490126,
            "range": "± 1985903",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46373002,
            "range": "± 1444388",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1220912103,
            "range": "± 18558112",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104823086,
            "range": "± 722168",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45749880,
            "range": "± 464614",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20739999,
            "range": "± 173667",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7747196,
            "range": "± 126850",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734071,
            "range": "± 11519",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733145,
            "range": "± 19012",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576818,
            "range": "± 20038",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 276022,
            "range": "± 5176",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 277517,
            "range": "± 3499",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 559871,
            "range": "± 9180",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 554644,
            "range": "± 3360",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1203308,
            "range": "± 10072",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1207997,
            "range": "± 9689",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2286546,
            "range": "± 15632",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2314317,
            "range": "± 25378",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4500928,
            "range": "± 75138",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4508393,
            "range": "± 52369",
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
          "id": "d3e0c904e713c5a576b424481c5dd651d2e3a557",
          "message": "avx512 eval at secure point (#513)",
          "timestamp": "2024-03-20T17:25:08+02:00",
          "tree_id": "eda576cd3c8bb642a67e0050a66fe0f98edc7db1",
          "url": "https://github.com/starkware-libs/stwo/commit/d3e0c904e713c5a576b424481c5dd651d2e3a557"
        },
        "date": 1710949259765,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76293960,
            "range": "± 676958",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 202172801,
            "range": "± 4604959",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16650909,
            "range": "± 119389",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211360890,
            "range": "± 1275857",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 261626974,
            "range": "± 2115774",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45723712,
            "range": "± 384356",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20892366,
            "range": "± 392618",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203824473,
            "range": "± 3658941",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46241096,
            "range": "± 969625",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1220056272,
            "range": "± 12413848",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105173498,
            "range": "± 1782917",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46060806,
            "range": "± 2699433",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20783551,
            "range": "± 203254",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7742629,
            "range": "± 67065",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733023,
            "range": "± 16385",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4742930,
            "range": "± 29912",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578606,
            "range": "± 8158",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 276568,
            "range": "± 4934",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 277678,
            "range": "± 9210",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 564179,
            "range": "± 7218",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 559156,
            "range": "± 3964",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1220946,
            "range": "± 13838",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 1219906,
            "range": "± 8359",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2340586,
            "range": "± 41922",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2306488,
            "range": "± 19374",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 4558153,
            "range": "± 30556",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4511093,
            "range": "± 180189",
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
          "id": "4c48ae4d3fe59c70c78f3813d48c209a54664c6e",
          "message": "Fix merkle benchmarks (#517)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/517)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T09:02:38+02:00",
          "tree_id": "a2c06aec1ce16c543ebcfd1b8da4407869409cd2",
          "url": "https://github.com/starkware-libs/stwo/commit/4c48ae4d3fe59c70c78f3813d48c209a54664c6e"
        },
        "date": 1711005117770,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76116636,
            "range": "± 1251496",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 201524415,
            "range": "± 2937108",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16385437,
            "range": "± 142505",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 213444592,
            "range": "± 2978575",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 262637811,
            "range": "± 1376506",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45760523,
            "range": "± 550964",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20786611,
            "range": "± 319656",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203976813,
            "range": "± 3949796",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46118953,
            "range": "± 700200",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218023081,
            "range": "± 7714095",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104997659,
            "range": "± 6293880",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45590397,
            "range": "± 347330",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20798506,
            "range": "± 558126",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7732059,
            "range": "± 97233",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4737093,
            "range": "± 11562",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728069,
            "range": "± 8100",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 575761,
            "range": "± 11366",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321805,
            "range": "± 3779",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 152105,
            "range": "± 2682",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 632090,
            "range": "± 12531",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 312924,
            "range": "± 10441",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1292017,
            "range": "± 11703",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 769248,
            "range": "± 12863",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2880031,
            "range": "± 27550",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1992463,
            "range": "± 70508",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5675733,
            "range": "± 129826",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3399013,
            "range": "± 89442",
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
          "id": "95612d26227be7600fa4b765ec192d9a8d1f06f9",
          "message": "FRI in Commitment Scheme (#476)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/476)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T09:27:19+02:00",
          "tree_id": "ab25a7d5a8e7e8094642950ce58d26debfdb30f0",
          "url": "https://github.com/starkware-libs/stwo/commit/95612d26227be7600fa4b765ec192d9a8d1f06f9"
        },
        "date": 1711006608939,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78474024,
            "range": "± 637865",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 215775527,
            "range": "± 4085275",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16848385,
            "range": "± 166567",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211286593,
            "range": "± 1160130",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 262316186,
            "range": "± 2872146",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45782379,
            "range": "± 406091",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20898719,
            "range": "± 531829",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203465300,
            "range": "± 4898612",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46177919,
            "range": "± 1041855",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1222271553,
            "range": "± 14132576",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104257203,
            "range": "± 3381574",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45713177,
            "range": "± 1238284",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20792064,
            "range": "± 398478",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7720690,
            "range": "± 82216",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733297,
            "range": "± 16713",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730332,
            "range": "± 26051",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576505,
            "range": "± 8809",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322427,
            "range": "± 7521",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 152828,
            "range": "± 6577",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 633252,
            "range": "± 7512",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 311322,
            "range": "± 8488",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1324337,
            "range": "± 32030",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 752730,
            "range": "± 27652",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2898748,
            "range": "± 90069",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2003649,
            "range": "± 90282",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5974733,
            "range": "± 86609",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3633751,
            "range": "± 55851",
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
          "id": "9079231dd0c058a889a31b9eef2df156da4a09d9",
          "message": "Revert \"integrated batch inverse in fib cpu (#448)\" (#477)\n\nThis reverts commit 3658aa094b49d1952b8d023d774e98343abd47e9.\n\n<!-- Reviewable:start -->\n- - -\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/477)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T09:32:55+02:00",
          "tree_id": "49682a01e8c9048ccb7c9290952b96ce14fdee2b",
          "url": "https://github.com/starkware-libs/stwo/commit/9079231dd0c058a889a31b9eef2df156da4a09d9"
        },
        "date": 1711006944885,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78011393,
            "range": "± 2378036",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 218051030,
            "range": "± 2282423",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16837022,
            "range": "± 259029",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211789284,
            "range": "± 3400994",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 266206808,
            "range": "± 1992097",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46165354,
            "range": "± 681730",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20737017,
            "range": "± 711679",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204176810,
            "range": "± 3658188",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46174566,
            "range": "± 347950",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214093029,
            "range": "± 15495336",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105134057,
            "range": "± 1333479",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45964507,
            "range": "± 499163",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20945430,
            "range": "± 456075",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7733424,
            "range": "± 79726",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4737367,
            "range": "± 17551",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4736839,
            "range": "± 11065",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577991,
            "range": "± 16705",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321656,
            "range": "± 2640",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 155932,
            "range": "± 3816",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 638482,
            "range": "± 25961",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 316707,
            "range": "± 11751",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1286424,
            "range": "± 11689",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 704907,
            "range": "± 30486",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2895409,
            "range": "± 21297",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1263555,
            "range": "± 9770",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5878789,
            "range": "± 71008",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3516465,
            "range": "± 81888",
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
          "id": "82fe65ec854e129c7e45210c16430bf5b153d9bb",
          "message": "allowed uneven slices in batch inverse (#519)",
          "timestamp": "2024-03-21T10:03:21+02:00",
          "tree_id": "35f7147b8429a9ef1257a8d1931218d76c7e1f7b",
          "url": "https://github.com/starkware-libs/stwo/commit/82fe65ec854e129c7e45210c16430bf5b153d9bb"
        },
        "date": 1711008876321,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78385920,
            "range": "± 2609300",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 237954299,
            "range": "± 14322949",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 17152952,
            "range": "± 508632",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211808664,
            "range": "± 1984851",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 283209662,
            "range": "± 11368170",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46206539,
            "range": "± 1441550",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20758419,
            "range": "± 453821",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204325607,
            "range": "± 1720003",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46024659,
            "range": "± 828890",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214747669,
            "range": "± 13480336",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105024133,
            "range": "± 2815562",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45904221,
            "range": "± 722651",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20748394,
            "range": "± 141013",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7729750,
            "range": "± 81111",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4741036,
            "range": "± 26496",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4737001,
            "range": "± 20514",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577840,
            "range": "± 13269",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321304,
            "range": "± 11546",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 152303,
            "range": "± 1827",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 620721,
            "range": "± 7495",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310565,
            "range": "± 7726",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1333192,
            "range": "± 31762",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 840008,
            "range": "± 19434",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2950250,
            "range": "± 126056",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1920576,
            "range": "± 57561",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6123887,
            "range": "± 144040",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3817558,
            "range": "± 55092",
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
          "id": "4ed11e94569063f511d93d3865483031934d186d",
          "message": "Use only canonic domains\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/463)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T13:24:07+02:00",
          "tree_id": "ab9c2be58691fec73cd75b25e495d7ae400c1872",
          "url": "https://github.com/starkware-libs/stwo/commit/4ed11e94569063f511d93d3865483031934d186d"
        },
        "date": 1711020813270,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77837217,
            "range": "± 1656746",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 236096791,
            "range": "± 4425619",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16908714,
            "range": "± 539141",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 212942186,
            "range": "± 1647439",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 268382882,
            "range": "± 5843054",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45757603,
            "range": "± 996808",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20895690,
            "range": "± 179527",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204245982,
            "range": "± 2084892",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46063168,
            "range": "± 1160732",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216897529,
            "range": "± 7880578",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105028030,
            "range": "± 906169",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45727639,
            "range": "± 962198",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20870609,
            "range": "± 181684",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7741476,
            "range": "± 54291",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735535,
            "range": "± 18593",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4727593,
            "range": "± 25632",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578185,
            "range": "± 26091",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 630,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323848,
            "range": "± 5581",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 155870,
            "range": "± 9972",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 623353,
            "range": "± 17326",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 312369,
            "range": "± 14854",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1330845,
            "range": "± 27917",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 832256,
            "range": "± 35351",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2932728,
            "range": "± 52190",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1283585,
            "range": "± 12860",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5831638,
            "range": "± 238812",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3961176,
            "range": "± 105173",
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
          "id": "c61780418786fa24b3ab27899a2e3f6cf9bbf850",
          "message": "Move SecureColumn\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/478)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T13:26:45+02:00",
          "tree_id": "85a111d9f5e90577683d9501b07a595233cd30d3",
          "url": "https://github.com/starkware-libs/stwo/commit/c61780418786fa24b3ab27899a2e3f6cf9bbf850"
        },
        "date": 1711020953244,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 71203774,
            "range": "± 680601",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 200004568,
            "range": "± 6922679",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16517702,
            "range": "± 180081",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211142457,
            "range": "± 3613139",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 262384474,
            "range": "± 1566741",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45593408,
            "range": "± 608338",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20834857,
            "range": "± 219524",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204431713,
            "range": "± 1695703",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46339512,
            "range": "± 681391",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215199004,
            "range": "± 9453919",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104991953,
            "range": "± 1981119",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45749967,
            "range": "± 448806",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20890647,
            "range": "± 822943",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7732008,
            "range": "± 50107",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4729306,
            "range": "± 13863",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4727637,
            "range": "± 8784",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577285,
            "range": "± 12658",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319319,
            "range": "± 4115",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 144901,
            "range": "± 3953",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 623808,
            "range": "± 7680",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 316651,
            "range": "± 4982",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1310952,
            "range": "± 13909",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 755579,
            "range": "± 45398",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2866404,
            "range": "± 39071",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1755353,
            "range": "± 24527",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5687031,
            "range": "± 163260",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3304789,
            "range": "± 57738",
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
          "id": "535b916d0af6b6d8fff3d19f883b7686aeb64417",
          "message": "Rename evalaution.rs to accumulaion.rs\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/479)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T13:30:35+02:00",
          "tree_id": "b43a281edd5f46f1d9c953415e5ca937035968d9",
          "url": "https://github.com/starkware-libs/stwo/commit/535b916d0af6b6d8fff3d19f883b7686aeb64417"
        },
        "date": 1711021192947,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 80762353,
            "range": "± 1501971",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 226875900,
            "range": "± 4235709",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16979593,
            "range": "± 184842",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211706615,
            "range": "± 2408564",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 274702642,
            "range": "± 1859687",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45616567,
            "range": "± 1543339",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20858288,
            "range": "± 278385",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204059322,
            "range": "± 3151720",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46154178,
            "range": "± 1017634",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216452210,
            "range": "± 15834312",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104900714,
            "range": "± 1655255",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45831834,
            "range": "± 809303",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20759417,
            "range": "± 192184",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7719931,
            "range": "± 52336",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733176,
            "range": "± 15986",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733431,
            "range": "± 10315",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577742,
            "range": "± 6153",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 318784,
            "range": "± 6327",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146331,
            "range": "± 3807",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 640246,
            "range": "± 11289",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 305552,
            "range": "± 7225",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1319064,
            "range": "± 35931",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 757307,
            "range": "± 7829",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2906920,
            "range": "± 32243",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1735285,
            "range": "± 19634",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6022084,
            "range": "± 57356",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3411985,
            "range": "± 30680",
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
          "id": "f482ba8d15951844d67c425d0de6e31a4f15470f",
          "message": "implemented batch inverse in secure field eval (#520)",
          "timestamp": "2024-03-21T13:50:06+02:00",
          "tree_id": "bf50a8b97aafe62bc08a4dc4c14e1eeaed35ce0f",
          "url": "https://github.com/starkware-libs/stwo/commit/f482ba8d15951844d67c425d0de6e31a4f15470f"
        },
        "date": 1711022430481,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75955156,
            "range": "± 847854",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 184526932,
            "range": "± 6628502",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16404551,
            "range": "± 222366",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 212092918,
            "range": "± 3606909",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 259524497,
            "range": "± 2833363",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45843231,
            "range": "± 646307",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21032568,
            "range": "± 603574",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203318206,
            "range": "± 1443648",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46150428,
            "range": "± 1033391",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1211220534,
            "range": "± 9502746",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105174295,
            "range": "± 950520",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45871830,
            "range": "± 1178912",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20804600,
            "range": "± 148698",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7722442,
            "range": "± 74477",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4732482,
            "range": "± 12154",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738907,
            "range": "± 29138",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576935,
            "range": "± 10688",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321090,
            "range": "± 6048",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147140,
            "range": "± 2508",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 628914,
            "range": "± 4202",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 315387,
            "range": "± 4986",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1305883,
            "range": "± 19904",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 804835,
            "range": "± 57953",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2882297,
            "range": "± 50717",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1773213,
            "range": "± 65878",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5671365,
            "range": "± 143939",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3375515,
            "range": "± 61407",
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
          "id": "e827cc9eb433fb6b877af46999d2ca9b5e16fdad",
          "message": "changed interface to mixed merkle tree\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/456)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-21T14:09:26+02:00",
          "tree_id": "677f3fe611b56c3fabde1e2198a99133c10a0744",
          "url": "https://github.com/starkware-libs/stwo/commit/e827cc9eb433fb6b877af46999d2ca9b5e16fdad"
        },
        "date": 1711023521101,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74601528,
            "range": "± 919722",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 225983789,
            "range": "± 7905161",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16749672,
            "range": "± 437997",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 214661624,
            "range": "± 8272010",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 248990890,
            "range": "± 3523302",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45903591,
            "range": "± 1236467",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20957055,
            "range": "± 443796",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203186656,
            "range": "± 43992720",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46671356,
            "range": "± 2848696",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214498712,
            "range": "± 10067392",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104396914,
            "range": "± 3815882",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46599168,
            "range": "± 2514719",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20610760,
            "range": "± 360975",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7718733,
            "range": "± 120549",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4727991,
            "range": "± 25079",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728548,
            "range": "± 7455",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576877,
            "range": "± 11567",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 758,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 324532,
            "range": "± 9148",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 150513,
            "range": "± 5292",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 612639,
            "range": "± 14733",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 306058,
            "range": "± 8253",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1303251,
            "range": "± 6371",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 710117,
            "range": "± 11057",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2850564,
            "range": "± 11938",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1268104,
            "range": "± 3404",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5801443,
            "range": "± 50418",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3237017,
            "range": "± 37631",
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
          "id": "3524e6b1d20264f626298fdde787238dddfb14e7",
          "message": "integrated mixed degree merkle tree (#515)",
          "timestamp": "2024-03-21T14:12:10+02:00",
          "tree_id": "26d146a8bcfd0535966310c5e53643eb4b437756",
          "url": "https://github.com/starkware-libs/stwo/commit/3524e6b1d20264f626298fdde787238dddfb14e7"
        },
        "date": 1711023738654,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 70374396,
            "range": "± 681866",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 220643688,
            "range": "± 3737801",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16738353,
            "range": "± 114331",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210878696,
            "range": "± 660814",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 262975916,
            "range": "± 2342760",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45541370,
            "range": "± 153832",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20616633,
            "range": "± 107851",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 202087086,
            "range": "± 2192680",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 45855919,
            "range": "± 585210",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1205314086,
            "range": "± 9364388",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 103951702,
            "range": "± 1519448",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45539342,
            "range": "± 610393",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20614219,
            "range": "± 104697",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7718400,
            "range": "± 37247",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736244,
            "range": "± 34312",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4756129,
            "range": "± 30715",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578527,
            "range": "± 15660",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 631,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 765,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 342913,
            "range": "± 29910",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146357,
            "range": "± 2552",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 632543,
            "range": "± 13719",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 306905,
            "range": "± 4848",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1323649,
            "range": "± 31689",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 863366,
            "range": "± 63146",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2625784,
            "range": "± 16467",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1284832,
            "range": "± 10574",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6203450,
            "range": "± 118217",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 2991787,
            "range": "± 64662",
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
          "id": "584c52da81a341f7da87af0628ad4e78a5da2ac4",
          "message": "DomainAccumulator allows accumulating a row at once (#480)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/480)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-24T15:17:13+02:00",
          "tree_id": "443675429d1198cdd5a3217076510701f1dd9713",
          "url": "https://github.com/starkware-libs/stwo/commit/584c52da81a341f7da87af0628ad4e78a5da2ac4"
        },
        "date": 1711286861125,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75704718,
            "range": "± 1353775",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 173883098,
            "range": "± 10219668",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16209501,
            "range": "± 195656",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211602238,
            "range": "± 4111446",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 248297848,
            "range": "± 1441067",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45616831,
            "range": "± 530580",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20786012,
            "range": "± 587384",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204091238,
            "range": "± 4499695",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46204691,
            "range": "± 623918",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1212808573,
            "range": "± 14792500",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105464918,
            "range": "± 2118300",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45752105,
            "range": "± 293910",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21005051,
            "range": "± 280498",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7719803,
            "range": "± 230541",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4729123,
            "range": "± 8984",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728786,
            "range": "± 10209",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 575644,
            "range": "± 7830",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 320016,
            "range": "± 3257",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145670,
            "range": "± 1118",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 629462,
            "range": "± 12494",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 307846,
            "range": "± 8461",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1271032,
            "range": "± 14778",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 686747,
            "range": "± 22562",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2877098,
            "range": "± 29275",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1272100,
            "range": "± 13408",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5696400,
            "range": "± 52215",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3264784,
            "range": "± 58154",
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
          "id": "20412e4a9f00507d771f24d66e4dbd59d0c66849",
          "message": "QuotientOps (#481)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/481)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-24T15:21:31+02:00",
          "tree_id": "9f05e76a99c2f7e87e654bf8d3be39bac2399078",
          "url": "https://github.com/starkware-libs/stwo/commit/20412e4a9f00507d771f24d66e4dbd59d0c66849"
        },
        "date": 1711287075944,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 82429296,
            "range": "± 1746520",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 240932413,
            "range": "± 6758296",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 17008223,
            "range": "± 118429",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 218992373,
            "range": "± 1697774",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 276780870,
            "range": "± 2927034",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45937323,
            "range": "± 813954",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21037190,
            "range": "± 346141",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204985424,
            "range": "± 3551068",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46373273,
            "range": "± 1905558",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218062923,
            "range": "± 14602384",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105492573,
            "range": "± 3190377",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45661910,
            "range": "± 211463",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20936733,
            "range": "± 734329",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7781241,
            "range": "± 89758",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4751731,
            "range": "± 20437",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4744327,
            "range": "± 9631",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578638,
            "range": "± 9027",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323964,
            "range": "± 4920",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 152213,
            "range": "± 3084",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 644338,
            "range": "± 12382",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 309946,
            "range": "± 7425",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1288954,
            "range": "± 11532",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 700258,
            "range": "± 35027",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2964770,
            "range": "± 36477",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1841123,
            "range": "± 70225",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6297567,
            "range": "± 109506",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4079743,
            "range": "± 96798",
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
          "id": "bcb5732959352735c3ae67dd9e29835fc6188e83",
          "message": "Separate commitment prover and verifier to modules (#482)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/482)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-25T06:09:55+02:00",
          "tree_id": "13e8057ab26c075b29ae13efd2939318e69162c2",
          "url": "https://github.com/starkware-libs/stwo/commit/bcb5732959352735c3ae67dd9e29835fc6188e83"
        },
        "date": 1711340500929,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 83384357,
            "range": "± 3015995",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 230950066,
            "range": "± 6611664",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 17089789,
            "range": "± 250543",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 218827060,
            "range": "± 4404700",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 278787442,
            "range": "± 7061376",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45868033,
            "range": "± 1276539",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20945365,
            "range": "± 669682",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205759851,
            "range": "± 2011475",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46502312,
            "range": "± 606800",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1221477357,
            "range": "± 14735647",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 106361769,
            "range": "± 2016913",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46034588,
            "range": "± 689791",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21014441,
            "range": "± 611213",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7760811,
            "range": "± 60100",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733249,
            "range": "± 16764",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4740447,
            "range": "± 11964",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577245,
            "range": "± 12015",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 324062,
            "range": "± 3136",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146575,
            "range": "± 11824",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 644473,
            "range": "± 5295",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 313028,
            "range": "± 7963",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1318492,
            "range": "± 33834",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 808275,
            "range": "± 32612",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 3018209,
            "range": "± 51816",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2044238,
            "range": "± 23345",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6290185,
            "range": "± 56691",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4042894,
            "range": "± 75039",
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
          "id": "3846f828a4a2f6059f1ca47f3c0c8b28038479f1",
          "message": "fft without copying (#535)",
          "timestamp": "2024-03-25T11:10:17+02:00",
          "tree_id": "5ca96be062c811d85b7ff55094516fd588c20702",
          "url": "https://github.com/starkware-libs/stwo/commit/3846f828a4a2f6059f1ca47f3c0c8b28038479f1"
        },
        "date": 1711358501619,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 80957898,
            "range": "± 1299163",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 224906894,
            "range": "± 3314216",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16730309,
            "range": "± 122468",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 213738616,
            "range": "± 4194545",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 268552484,
            "range": "± 1935252",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3449450,
            "range": "± 33137",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45633911,
            "range": "± 218746",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20963438,
            "range": "± 289169",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204783651,
            "range": "± 3965369",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46188469,
            "range": "± 477839",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1220296234,
            "range": "± 16798861",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105273925,
            "range": "± 5099486",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45721719,
            "range": "± 171042",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20767957,
            "range": "± 448483",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7799835,
            "range": "± 112425",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736156,
            "range": "± 10024",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4739377,
            "range": "± 10880",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577766,
            "range": "± 12609",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 31",
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
            "value": 322259,
            "range": "± 4036",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146562,
            "range": "± 1005",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 626426,
            "range": "± 14046",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 308987,
            "range": "± 9933",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1320632,
            "range": "± 28455",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 755382,
            "range": "± 23299",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2920281,
            "range": "± 32721",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1772224,
            "range": "± 58311",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6042361,
            "range": "± 30604",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3736870,
            "range": "± 105753",
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
          "id": "83002ed07f9b049b3fa0b319acf105d243d482b6",
          "message": "Use fast eval_at_point (#536)",
          "timestamp": "2024-03-25T11:54:26+02:00",
          "tree_id": "2580a9574026c7ae59869963773c64decdcf7f4e",
          "url": "https://github.com/starkware-libs/stwo/commit/83002ed07f9b049b3fa0b319acf105d243d482b6"
        },
        "date": 1711361118078,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74924499,
            "range": "± 685850",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 181738130,
            "range": "± 7924982",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16239382,
            "range": "± 128708",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211590549,
            "range": "± 3609761",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 268799570,
            "range": "± 1641534",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3225812,
            "range": "± 42880",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45585002,
            "range": "± 334258",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20916819,
            "range": "± 302991",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 202859479,
            "range": "± 3465606",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46083000,
            "range": "± 651731",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217286261,
            "range": "± 10099387",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104836791,
            "range": "± 4081303",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45567477,
            "range": "± 195151",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20674796,
            "range": "± 418726",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7760223,
            "range": "± 149867",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4729768,
            "range": "± 12478",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4724761,
            "range": "± 11594",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577394,
            "range": "± 10794",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321997,
            "range": "± 8283",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145851,
            "range": "± 916",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 636242,
            "range": "± 15339",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 313261,
            "range": "± 7841",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1323460,
            "range": "± 21557",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 806369,
            "range": "± 11105",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2932858,
            "range": "± 82480",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1916452,
            "range": "± 68136",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5959925,
            "range": "± 131884",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3772518,
            "range": "± 91938",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8a75935de47446049553bf31b88a2c99389282c5",
          "message": "Support mixed degree in commitment scheme. (#533)",
          "timestamp": "2024-03-25T14:12:56+02:00",
          "tree_id": "782b322f3327abfa5f9c51c020bb0f7678096cdf",
          "url": "https://github.com/starkware-libs/stwo/commit/8a75935de47446049553bf31b88a2c99389282c5"
        },
        "date": 1711369332180,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78744902,
            "range": "± 2045396",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 225328903,
            "range": "± 5299393",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16543597,
            "range": "± 237669",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 212047749,
            "range": "± 5412066",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 248442884,
            "range": "± 1355676",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3234170,
            "range": "± 29718",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45590989,
            "range": "± 237426",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21269374,
            "range": "± 789734",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204355945,
            "range": "± 3937767",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46290674,
            "range": "± 2726967",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1209953522,
            "range": "± 13067197",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104176207,
            "range": "± 1702756",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45459066,
            "range": "± 183266",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20604902,
            "range": "± 103854",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7710970,
            "range": "± 48005",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4732040,
            "range": "± 15949",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4727342,
            "range": "± 29081",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 575419,
            "range": "± 5906",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 622,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 751,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319200,
            "range": "± 5091",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146863,
            "range": "± 1874",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 629307,
            "range": "± 25563",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310788,
            "range": "± 17465",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1309427,
            "range": "± 23972",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 725851,
            "range": "± 17437",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2906759,
            "range": "± 42741",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1739673,
            "range": "± 37876",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5743628,
            "range": "± 161537",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3566852,
            "range": "± 25825",
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
          "id": "5e418eeb5efac5b0714b7d6d872c33046d17a789",
          "message": "Prepare commitmentschem per size (#537)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/537)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-25T15:05:35+02:00",
          "tree_id": "9fbda2353b69195f47c79e887dc07f14bc8e33bc",
          "url": "https://github.com/starkware-libs/stwo/commit/5e418eeb5efac5b0714b7d6d872c33046d17a789"
        },
        "date": 1711372512864,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77364151,
            "range": "± 1196841",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 222243202,
            "range": "± 3986221",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16645342,
            "range": "± 97262",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211654214,
            "range": "± 1658444",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 269742713,
            "range": "± 4251585",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3307540,
            "range": "± 81211",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45654721,
            "range": "± 579586",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20788759,
            "range": "± 174043",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204220710,
            "range": "± 4850356",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46313702,
            "range": "± 1419029",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215553880,
            "range": "± 15604604",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105675477,
            "range": "± 2899665",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45779490,
            "range": "± 396474",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20779662,
            "range": "± 82201",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7723374,
            "range": "± 93972",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734585,
            "range": "± 11091",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4732896,
            "range": "± 13161",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 575702,
            "range": "± 10809",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319433,
            "range": "± 5788",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 149954,
            "range": "± 9529",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 624302,
            "range": "± 7290",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 315325,
            "range": "± 10401",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1295420,
            "range": "± 54012",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 707903,
            "range": "± 18455",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2904596,
            "range": "± 116607",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1833062,
            "range": "± 46230",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5960554,
            "range": "± 80843",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3469038,
            "range": "± 59868",
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
          "id": "60d1432165bba9c42d851a1f68766a4588e1953d",
          "message": "CI on macos (#538)",
          "timestamp": "2024-03-25T15:12:19+02:00",
          "tree_id": "9fc5ac0e0afe5d94346a774c6fc197c4de965e25",
          "url": "https://github.com/starkware-libs/stwo/commit/60d1432165bba9c42d851a1f68766a4588e1953d"
        },
        "date": 1711372913778,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73086440,
            "range": "± 694188",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 194380258,
            "range": "± 1554787",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16795867,
            "range": "± 127650",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211202936,
            "range": "± 4027520",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 263636843,
            "range": "± 1403169",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3284141,
            "range": "± 22850",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45813734,
            "range": "± 436862",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20897921,
            "range": "± 739301",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203268775,
            "range": "± 4446524",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46174871,
            "range": "± 347826",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213322381,
            "range": "± 12711798",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 106306022,
            "range": "± 5317870",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45641852,
            "range": "± 200555",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20824019,
            "range": "± 975864",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7731015,
            "range": "± 48948",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4732321,
            "range": "± 36323",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728721,
            "range": "± 11287",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577165,
            "range": "± 9006",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 629,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319549,
            "range": "± 1956",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 149573,
            "range": "± 4160",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 631007,
            "range": "± 10755",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 311683,
            "range": "± 13459",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1314600,
            "range": "± 3780",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 780270,
            "range": "± 13623",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2905124,
            "range": "± 39026",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1823612,
            "range": "± 43093",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5935662,
            "range": "± 131849",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3413112,
            "range": "± 27042",
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
          "id": "9dbe5bc23d226414d9878d1a3aea63a546771fff",
          "message": "avx blake (#518)",
          "timestamp": "2024-03-25T16:27:17+02:00",
          "tree_id": "d786e6ab0f8cf3bd2c18c2ebb977180024f5d04e",
          "url": "https://github.com/starkware-libs/stwo/commit/9dbe5bc23d226414d9878d1a3aea63a546771fff"
        },
        "date": 1711377514852,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73399779,
            "range": "± 1369014",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 190784691,
            "range": "± 4806164",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16805282,
            "range": "± 101388",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211180759,
            "range": "± 2915310",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 255592091,
            "range": "± 1844733",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3284265,
            "range": "± 53755",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46077924,
            "range": "± 1902834",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20830558,
            "range": "± 549424",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204835055,
            "range": "± 3554215",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46058079,
            "range": "± 412271",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219806838,
            "range": "± 12785527",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105271464,
            "range": "± 931840",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45862268,
            "range": "± 421377",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20761498,
            "range": "± 838223",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7739813,
            "range": "± 38690",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736464,
            "range": "± 13520",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729119,
            "range": "± 11696",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576767,
            "range": "± 11637",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 320163,
            "range": "± 7882",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148188,
            "range": "± 3441",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 625694,
            "range": "± 15720",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 306288,
            "range": "± 4303",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1322236,
            "range": "± 20108",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 711782,
            "range": "± 17002",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2873137,
            "range": "± 44233",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1769355,
            "range": "± 19291",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5914571,
            "range": "± 140170",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3372510,
            "range": "± 39132",
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
          "id": "1ebc2beeec84a8255fd49aee41507f6f65bfa4e1",
          "message": "Simple mixed merkle tree (#525)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/525)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-26T09:16:43+02:00",
          "tree_id": "a7a4bfa5390e4daf58f0ccd4cf5ee28a27044ae1",
          "url": "https://github.com/starkware-libs/stwo/commit/1ebc2beeec84a8255fd49aee41507f6f65bfa4e1"
        },
        "date": 1711437980644,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75468541,
            "range": "± 1315240",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 208644451,
            "range": "± 2650474",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16671898,
            "range": "± 108500",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210579054,
            "range": "± 1817387",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 261827549,
            "range": "± 1948563",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3279224,
            "range": "± 39800",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45964092,
            "range": "± 277151",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20854793,
            "range": "± 218757",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203518903,
            "range": "± 4833262",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46114566,
            "range": "± 1377048",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1221399412,
            "range": "± 15955047",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105435624,
            "range": "± 2185319",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45659868,
            "range": "± 573457",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20755993,
            "range": "± 330188",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7729207,
            "range": "± 63489",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4741358,
            "range": "± 16206",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738874,
            "range": "± 7867",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577802,
            "range": "± 7937",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322124,
            "range": "± 14004",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 152490,
            "range": "± 4875",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 628620,
            "range": "± 4023",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 307750,
            "range": "± 11102",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1311540,
            "range": "± 38534",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 775292,
            "range": "± 29124",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2902972,
            "range": "± 138618",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1830178,
            "range": "± 43475",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5943668,
            "range": "± 79543",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3381974,
            "range": "± 81237",
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
          "id": "dd8770d8c4d48fe61b5bf8686316504bd67d1c02",
          "message": "Test docs in CI (#539)",
          "timestamp": "2024-03-26T09:32:02+02:00",
          "tree_id": "453c25b7dceb4dfd61c0614e5d8221324e37e731",
          "url": "https://github.com/starkware-libs/stwo/commit/dd8770d8c4d48fe61b5bf8686316504bd67d1c02"
        },
        "date": 1711438960605,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 71745835,
            "range": "± 664747",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 193659185,
            "range": "± 3213178",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16416758,
            "range": "± 236932",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 212075326,
            "range": "± 3459878",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 258470926,
            "range": "± 2375286",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3234510,
            "range": "± 34874",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45807655,
            "range": "± 1179930",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20753909,
            "range": "± 434591",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204655478,
            "range": "± 3840505",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46263394,
            "range": "± 1570821",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214296169,
            "range": "± 12874660",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105204017,
            "range": "± 1929181",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45642104,
            "range": "± 392241",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20826094,
            "range": "± 497629",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7744736,
            "range": "± 101958",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4756174,
            "range": "± 45147",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4725368,
            "range": "± 11824",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577092,
            "range": "± 7379",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 758,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322973,
            "range": "± 5358",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146740,
            "range": "± 1380",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 640331,
            "range": "± 15763",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 316774,
            "range": "± 7618",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1332083,
            "range": "± 18034",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 829391,
            "range": "± 21065",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2915752,
            "range": "± 43916",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1891875,
            "range": "± 48470",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5734357,
            "range": "± 50783",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3475370,
            "range": "± 93342",
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
          "id": "a7e8bd14b9b676995eb9fabe0c96ce18650adbff",
          "message": "Commitment Scheme evaluation per size (#483)",
          "timestamp": "2024-03-26T11:00:05+02:00",
          "tree_id": "4193c97d73be6c1c991143b4b5bc94334ffbdf3a",
          "url": "https://github.com/starkware-libs/stwo/commit/a7e8bd14b9b676995eb9fabe0c96ce18650adbff"
        },
        "date": 1711444286562,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 70259181,
            "range": "± 1063183",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 216593982,
            "range": "± 3085939",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16800255,
            "range": "± 234765",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211819755,
            "range": "± 4209426",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 267731993,
            "range": "± 2193667",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3307088,
            "range": "± 32839",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45673448,
            "range": "± 834104",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20836770,
            "range": "± 610383",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204447695,
            "range": "± 4795108",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46014328,
            "range": "± 1119094",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214793779,
            "range": "± 15555773",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105384492,
            "range": "± 825620",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45881649,
            "range": "± 390259",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20948380,
            "range": "± 288992",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7713919,
            "range": "± 47185",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4738410,
            "range": "± 28736",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4732432,
            "range": "± 17748",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576937,
            "range": "± 10878",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 90",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 324343,
            "range": "± 3262",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146011,
            "range": "± 2009",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 626448,
            "range": "± 16647",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310887,
            "range": "± 10832",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1331275,
            "range": "± 8061",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 818884,
            "range": "± 43604",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2648759,
            "range": "± 44562",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1283074,
            "range": "± 25899",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5882712,
            "range": "± 154656",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3607980,
            "range": "± 45307",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "822f10d8d7cf7ddd12c117b4373f41f38d70d7dc",
          "message": "Organize errors. (#523)",
          "timestamp": "2024-03-26T11:48:21+02:00",
          "tree_id": "25747556baea8daaf3f505e23a9ce8c2e55203e2",
          "url": "https://github.com/starkware-libs/stwo/commit/822f10d8d7cf7ddd12c117b4373f41f38d70d7dc"
        },
        "date": 1711447154963,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 82252876,
            "range": "± 1561884",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 232630121,
            "range": "± 3262300",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16928205,
            "range": "± 122195",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211262796,
            "range": "± 4494169",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 275707328,
            "range": "± 2936481",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3454914,
            "range": "± 51274",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45866576,
            "range": "± 651276",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20897643,
            "range": "± 827360",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204035006,
            "range": "± 4026390",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46265856,
            "range": "± 321670",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217472074,
            "range": "± 14342695",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105076704,
            "range": "± 1964716",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45602680,
            "range": "± 643746",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20826446,
            "range": "± 628301",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7772920,
            "range": "± 148086",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4730166,
            "range": "± 11401",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4741396,
            "range": "± 18257",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577606,
            "range": "± 9710",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319581,
            "range": "± 4314",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145403,
            "range": "± 3556",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 640134,
            "range": "± 20026",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 319242,
            "range": "± 9346",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1313058,
            "range": "± 14677",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 736331,
            "range": "± 14171",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2949815,
            "range": "± 52283",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1748524,
            "range": "± 31491",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6109462,
            "range": "± 75053",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3678262,
            "range": "± 123367",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8694c9aa784ef2caec9388d951371c1fa1678b8c",
          "message": "Create example airs folder (#540)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/540)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-27T14:10:59+02:00",
          "tree_id": "9ec2f6519e27690c6e81be7966d5e35d4c68c109",
          "url": "https://github.com/starkware-libs/stwo/commit/8694c9aa784ef2caec9388d951371c1fa1678b8c"
        },
        "date": 1711542125528,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73596396,
            "range": "± 904982",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 122010129,
            "range": "± 8383775",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 15759247,
            "range": "± 82091",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210335061,
            "range": "± 5330376",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 256427050,
            "range": "± 1142450",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3155512,
            "range": "± 30495",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45705653,
            "range": "± 183477",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20905068,
            "range": "± 465790",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203453924,
            "range": "± 3236320",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46376990,
            "range": "± 1626616",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214601766,
            "range": "± 11502451",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105505420,
            "range": "± 2291712",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45745681,
            "range": "± 267277",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20766457,
            "range": "± 170869",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7714200,
            "range": "± 98480",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4730330,
            "range": "± 12342",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4725822,
            "range": "± 12697",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576478,
            "range": "± 7203",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 318102,
            "range": "± 11947",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147316,
            "range": "± 627",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 628074,
            "range": "± 6118",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 304664,
            "range": "± 5109",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1311144,
            "range": "± 24142",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 740745,
            "range": "± 10672",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2849551,
            "range": "± 46676",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1650101,
            "range": "± 30627",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5644259,
            "range": "± 136913",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3185486,
            "range": "± 21611",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a157d7673e39f767e6639e20de003edf9903cf98",
          "message": "Create wide fib structure (#541)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/541)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-27T14:17:40+02:00",
          "tree_id": "9d3665c27c5b8263a5ae6f1f89ba36864f7e130f",
          "url": "https://github.com/starkware-libs/stwo/commit/a157d7673e39f767e6639e20de003edf9903cf98"
        },
        "date": 1711542438594,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77610323,
            "range": "± 874294",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 224538509,
            "range": "± 4010450",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16697278,
            "range": "± 93514",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211072030,
            "range": "± 3073226",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 266340628,
            "range": "± 4591062",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3258855,
            "range": "± 60008",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45826575,
            "range": "± 373866",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20774284,
            "range": "± 195763",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203836835,
            "range": "± 3803611",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46317565,
            "range": "± 1444787",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1224967164,
            "range": "± 17587638",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105022575,
            "range": "± 1628580",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46070684,
            "range": "± 1667728",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20754571,
            "range": "± 98984",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7725400,
            "range": "± 64679",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4738232,
            "range": "± 16368",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4746622,
            "range": "± 23333",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578856,
            "range": "± 19679",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321586,
            "range": "± 10089",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146580,
            "range": "± 1963",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 628090,
            "range": "± 5062",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 314513,
            "range": "± 9815",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1334122,
            "range": "± 21214",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 768429,
            "range": "± 38757",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2946558,
            "range": "± 76458",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2040685,
            "range": "± 45513",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5952389,
            "range": "± 90711",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3711341,
            "range": "± 79839",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4e331fce380cedbbf51a30ed1fcd7fa49939ed00",
          "message": "Write trace and trace asserts (#542)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/542)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-27T14:22:30+02:00",
          "tree_id": "a499742abe4c24fcf0cc4cf467ccf9c612948456",
          "url": "https://github.com/starkware-libs/stwo/commit/4e331fce380cedbbf51a30ed1fcd7fa49939ed00"
        },
        "date": 1711542713842,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73598612,
            "range": "± 884388",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 121764180,
            "range": "± 2044079",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 15703868,
            "range": "± 165975",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211854479,
            "range": "± 3988126",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 255495082,
            "range": "± 1991068",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3148487,
            "range": "± 21331",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45691876,
            "range": "± 977981",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20776233,
            "range": "± 482664",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205393757,
            "range": "± 3984824",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46044175,
            "range": "± 322478",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213688750,
            "range": "± 10195194",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104743913,
            "range": "± 933401",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45537500,
            "range": "± 518869",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20818300,
            "range": "± 693984",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7715631,
            "range": "± 60595",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4721879,
            "range": "± 5382",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729376,
            "range": "± 12596",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576719,
            "range": "± 15135",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 318057,
            "range": "± 1861",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 153430,
            "range": "± 2792",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 632507,
            "range": "± 20131",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 302259,
            "range": "± 3999",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1320070,
            "range": "± 5491",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 661945,
            "range": "± 8683",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2833641,
            "range": "± 53706",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1681140,
            "range": "± 17685",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5653188,
            "range": "± 200917",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3163602,
            "range": "± 48576",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8210ed0eb8bc7160b85e4f61174dbbb198bf7758",
          "message": "Constraints eval for wide fib (#543)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/543)\n<!-- Reviewable:end -->",
          "timestamp": "2024-03-27T14:26:39+02:00",
          "tree_id": "3302e071ea132a4030334967ccf85ca6dac75015",
          "url": "https://github.com/starkware-libs/stwo/commit/8210ed0eb8bc7160b85e4f61174dbbb198bf7758"
        },
        "date": 1711542998602,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 85017443,
            "range": "± 1697863",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 256264445,
            "range": "± 8078096",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 17159610,
            "range": "± 402301",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 212160081,
            "range": "± 3000563",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 288212285,
            "range": "± 5153836",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3495985,
            "range": "± 43808",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45667377,
            "range": "± 305777",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20877206,
            "range": "± 268899",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203763101,
            "range": "± 4165209",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46241744,
            "range": "± 1134278",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218505611,
            "range": "± 14991607",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 106231863,
            "range": "± 2381228",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45935946,
            "range": "± 3883606",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20991089,
            "range": "± 271098",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7731176,
            "range": "± 100919",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4742433,
            "range": "± 27664",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4750391,
            "range": "± 34109",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 579333,
            "range": "± 7866",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 324291,
            "range": "± 10198",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148652,
            "range": "± 2800",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 635295,
            "range": "± 11529",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 314502,
            "range": "± 13643",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1302954,
            "range": "± 161168",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 796163,
            "range": "± 45499",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2629433,
            "range": "± 34465",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1281713,
            "range": "± 10297",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6393321,
            "range": "± 171265",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4245302,
            "range": "± 67765",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d95d7422d688f94cd1d5208405062fd04e367320",
          "message": "Semantic quotient changes. (#544)",
          "timestamp": "2024-03-27T16:46:38+02:00",
          "tree_id": "43d1b77c756187264594fc2077224a778975cb65",
          "url": "https://github.com/starkware-libs/stwo/commit/d95d7422d688f94cd1d5208405062fd04e367320"
        },
        "date": 1711551481446,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76598138,
            "range": "± 844637",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 197813287,
            "range": "± 3496788",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16445539,
            "range": "± 111209",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210934684,
            "range": "± 1626058",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 263845592,
            "range": "± 1611180",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3194639,
            "range": "± 30556",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46540420,
            "range": "± 2466801",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20797330,
            "range": "± 108663",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203315253,
            "range": "± 3553328",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46231266,
            "range": "± 1121829",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216974353,
            "range": "± 13088132",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105042376,
            "range": "± 2248568",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45704237,
            "range": "± 1399293",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20826649,
            "range": "± 664941",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7711955,
            "range": "± 125398",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4730345,
            "range": "± 18008",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4737259,
            "range": "± 21347",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578438,
            "range": "± 6872",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323737,
            "range": "± 6667",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146598,
            "range": "± 2988",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 636276,
            "range": "± 6878",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 309695,
            "range": "± 7318",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1336800,
            "range": "± 32625",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 742197,
            "range": "± 26781",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2939013,
            "range": "± 53379",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2001204,
            "range": "± 94977",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5772189,
            "range": "± 76566",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3594723,
            "range": "± 59380",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "50c8a203d1853844521c0485f9f78e142b0967cb",
          "message": "Implement complex_conjugate_line_constants. (#545)",
          "timestamp": "2024-03-27T16:52:28+02:00",
          "tree_id": "ec48ea85a2e7f082714e1de3ab1bab996fe10327",
          "url": "https://github.com/starkware-libs/stwo/commit/50c8a203d1853844521c0485f9f78e142b0967cb"
        },
        "date": 1711551715007,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76347657,
            "range": "± 905338",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 209205353,
            "range": "± 5618277",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16604359,
            "range": "± 235868",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210251525,
            "range": "± 1017578",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 265629949,
            "range": "± 1231002",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3189434,
            "range": "± 46238",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45662203,
            "range": "± 269531",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20788962,
            "range": "± 331947",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 202784464,
            "range": "± 3611483",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46140658,
            "range": "± 1483840",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216662985,
            "range": "± 17942574",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105260538,
            "range": "± 2892449",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45651169,
            "range": "± 294275",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20745001,
            "range": "± 183245",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7702614,
            "range": "± 79105",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4726885,
            "range": "± 16363",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730350,
            "range": "± 10117",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577028,
            "range": "± 10593",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 752,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319025,
            "range": "± 12349",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148873,
            "range": "± 695",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 626476,
            "range": "± 9203",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 307966,
            "range": "± 2830",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1282521,
            "range": "± 42226",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 722097,
            "range": "± 24999",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2606019,
            "range": "± 27067",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1261094,
            "range": "± 5309",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5685334,
            "range": "± 81974",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3244661,
            "range": "± 39347",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6594763be056a02f4514b2acbc3bc783b3c801e3",
          "message": "Precompute complex conjugate line constants. (#546)",
          "timestamp": "2024-03-27T16:57:27+02:00",
          "tree_id": "08ca43cb8656fa30d1970063ee7b6797dce1bd42",
          "url": "https://github.com/starkware-libs/stwo/commit/6594763be056a02f4514b2acbc3bc783b3c801e3"
        },
        "date": 1711552021669,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75463745,
            "range": "± 1005305",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 165468749,
            "range": "± 5686137",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16083615,
            "range": "± 266442",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211738504,
            "range": "± 2910698",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 258008200,
            "range": "± 3461132",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3167888,
            "range": "± 27305",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45676031,
            "range": "± 804935",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20726989,
            "range": "± 602277",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203710980,
            "range": "± 2185177",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46537198,
            "range": "± 927529",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216747783,
            "range": "± 9732335",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105356358,
            "range": "± 771336",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45568014,
            "range": "± 553183",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20897980,
            "range": "± 294484",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7715953,
            "range": "± 46636",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4727894,
            "range": "± 20996",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4731330,
            "range": "± 26635",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576935,
            "range": "± 14121",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319467,
            "range": "± 9623",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148210,
            "range": "± 1885",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 624410,
            "range": "± 6343",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 312108,
            "range": "± 5568",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1323093,
            "range": "± 22007",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 791267,
            "range": "± 38307",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2895716,
            "range": "± 50798",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1698645,
            "range": "± 39430",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5646952,
            "range": "± 107577",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3314301,
            "range": "± 65351",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "136858ab44b23b5831219e5f6a7ba15b3cbca768",
          "message": "Remove oods file. (#547)",
          "timestamp": "2024-03-27T17:01:41+02:00",
          "tree_id": "517978177b5e5070c2bbb77d4a7e298e408bab13",
          "url": "https://github.com/starkware-libs/stwo/commit/136858ab44b23b5831219e5f6a7ba15b3cbca768"
        },
        "date": 1711552272855,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75294658,
            "range": "± 571637",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 196457583,
            "range": "± 3009778",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16414837,
            "range": "± 66837",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211303122,
            "range": "± 3535121",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 266604169,
            "range": "± 1515509",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3193698,
            "range": "± 28097",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45512587,
            "range": "± 192264",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20686683,
            "range": "± 265597",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204184739,
            "range": "± 2808529",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46265617,
            "range": "± 539241",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219800261,
            "range": "± 11926819",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105831294,
            "range": "± 2862973",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45918099,
            "range": "± 765357",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20730364,
            "range": "± 361951",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7715170,
            "range": "± 135931",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4730248,
            "range": "± 4422",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729064,
            "range": "± 7189",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577586,
            "range": "± 6734",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322916,
            "range": "± 3231",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 153637,
            "range": "± 4210",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 633877,
            "range": "± 8001",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310279,
            "range": "± 5720",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1327800,
            "range": "± 15168",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 877142,
            "range": "± 36333",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2897164,
            "range": "± 47183",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1271059,
            "range": "± 11553",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5731645,
            "range": "± 59827",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3606421,
            "range": "± 54250",
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
          "id": "d4ebbc6b40ff3f39dd09f28a9e894618984278b0",
          "message": "New AVX quotients (#549)",
          "timestamp": "2024-03-28T14:12:16Z",
          "tree_id": "3f35271e22c624a55041de6e83908a95bf93c9ad",
          "url": "https://github.com/starkware-libs/stwo/commit/d4ebbc6b40ff3f39dd09f28a9e894618984278b0"
        },
        "date": 1711635806863,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 68018747,
            "range": "± 785345",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 210910805,
            "range": "± 4192047",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16621357,
            "range": "± 62012",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 211218828,
            "range": "± 2964771",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 266990102,
            "range": "± 2037046",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3243244,
            "range": "± 26333",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45666544,
            "range": "± 417145",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20791676,
            "range": "± 277037",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203446107,
            "range": "± 5239869",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46300289,
            "range": "± 1082631",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214842089,
            "range": "± 14729224",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104912386,
            "range": "± 3246266",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45600078,
            "range": "± 271016",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20729373,
            "range": "± 390696",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7703746,
            "range": "± 132023",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4752378,
            "range": "± 37593",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4734338,
            "range": "± 23891",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576515,
            "range": "± 7211",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323828,
            "range": "± 5996",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 150995,
            "range": "± 13328",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 638967,
            "range": "± 9335",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 319389,
            "range": "± 9873",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1290344,
            "range": "± 9383",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 692686,
            "range": "± 22890",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2883901,
            "range": "± 55516",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1262807,
            "range": "± 10667",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5923667,
            "range": "± 75394",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3623149,
            "range": "± 78942",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 3658985696,
            "range": "± 18165565",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 650633395,
            "range": "± 7511801",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2a18eb8f09b5e70ed4c10c918e8152bc6981644f",
          "message": "Optimize AVX quotienting. (#555)",
          "timestamp": "2024-03-28T17:17:21+02:00",
          "tree_id": "bbf5e5357f6eb0718e4d499c3af9d4232a2029ed",
          "url": "https://github.com/starkware-libs/stwo/commit/2a18eb8f09b5e70ed4c10c918e8152bc6981644f"
        },
        "date": 1711639657017,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 71855187,
            "range": "± 512806",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 167049253,
            "range": "± 4975689",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point",
            "value": 16377032,
            "range": "± 163482",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point",
            "value": 210829455,
            "range": "± 3433719",
            "unit": "ns/iter"
          },
          {
            "name": "avx ifft 26bit",
            "value": 257575166,
            "range": "± 1949822",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3174470,
            "range": "± 13416",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45798142,
            "range": "± 931091",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20721193,
            "range": "± 733973",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203298915,
            "range": "± 1363848",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46200216,
            "range": "± 515576",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217265085,
            "range": "± 11387865",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105169398,
            "range": "± 1249103",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45920356,
            "range": "± 841290",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20823261,
            "range": "± 469807",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7722863,
            "range": "± 49806",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4742652,
            "range": "± 16888",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728912,
            "range": "± 36223",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577445,
            "range": "± 12975",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323021,
            "range": "± 4376",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147341,
            "range": "± 1585",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 626829,
            "range": "± 16535",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 323081,
            "range": "± 16147",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1317412,
            "range": "± 14907",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 864943,
            "range": "± 25573",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2917790,
            "range": "± 72383",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1960347,
            "range": "± 66785",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5731504,
            "range": "± 98998",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3504414,
            "range": "± 74253",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1843901491,
            "range": "± 14090567",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 648635009,
            "range": "± 11285799",
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
          "id": "df3b5d1c6c107b8e9b990a7eb25af2c09e8fdf35",
          "message": "More detailed fft benchmarks (#532)",
          "timestamp": "2024-03-31T18:55:43+11:00",
          "tree_id": "01b94a691cd06aea147d8161ae2a50f6ceb0b5fe",
          "url": "https://github.com/starkware-libs/stwo/commit/df3b5d1c6c107b8e9b990a7eb25af2c09e8fdf35"
        },
        "date": 1711872659856,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 67296726,
            "range": "± 1117684",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 152790377,
            "range": "± 7332505",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1966114,
            "range": "± 42394",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26503810,
            "range": "± 773480",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 114048,
            "range": "± 1000",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267981,
            "range": "± 1525",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 565214,
            "range": "± 3532",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1192897,
            "range": "± 10720",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2476463,
            "range": "± 25810",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5121509,
            "range": "± 30479",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10934532,
            "range": "± 59848",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 23682622,
            "range": "± 282984",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 55716481,
            "range": "± 459012",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 119116191,
            "range": "± 2145181",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 252578890,
            "range": "± 2557444",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 538973868,
            "range": "± 2870740",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1223287018,
            "range": "± 10214813",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12312,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4481,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 307641,
            "range": "± 3921",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3182943,
            "range": "± 10179",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45688591,
            "range": "± 390080",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20793160,
            "range": "± 307731",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203060901,
            "range": "± 3428610",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46171872,
            "range": "± 821144",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213611846,
            "range": "± 12298090",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104427080,
            "range": "± 2986693",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45571511,
            "range": "± 277226",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20792750,
            "range": "± 327993",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7725643,
            "range": "± 79250",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4731141,
            "range": "± 88947",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4732542,
            "range": "± 15272",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577094,
            "range": "± 5279",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321538,
            "range": "± 3069",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147760,
            "range": "± 2982",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 631744,
            "range": "± 8864",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 304572,
            "range": "± 10087",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1295351,
            "range": "± 9287",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 762953,
            "range": "± 30934",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2892042,
            "range": "± 25094",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1710705,
            "range": "± 27386",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5642929,
            "range": "± 66561",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3263053,
            "range": "± 21744",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1861381119,
            "range": "± 16271153",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 648717544,
            "range": "± 5357154",
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
          "id": "bb5e4f350689549768350405d3e78a3022b76b2f",
          "message": "merkle intermediate layers (#548)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/548)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-02T14:18:30+03:00",
          "tree_id": "bc623446227e6648d7a20a89bab41d5ccd517d41",
          "url": "https://github.com/starkware-libs/stwo/commit/bb5e4f350689549768350405d3e78a3022b76b2f"
        },
        "date": 1712057613310,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77914010,
            "range": "± 2664765",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 230565779,
            "range": "± 15211559",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1980147,
            "range": "± 22881",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26440157,
            "range": "± 208102",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 114008,
            "range": "± 1377",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267275,
            "range": "± 3047",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 567256,
            "range": "± 8908",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1186431,
            "range": "± 6034",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2490027,
            "range": "± 13430",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5459123,
            "range": "± 192644",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12576354,
            "range": "± 538730",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 29295777,
            "range": "± 580628",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 62554678,
            "range": "± 2519676",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 129174664,
            "range": "± 2083577",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 272579036,
            "range": "± 10350825",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 521803875,
            "range": "± 15459585",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1271268749,
            "range": "± 30359943",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12380,
            "range": "± 123",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4478,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 324058,
            "range": "± 3464",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3349644,
            "range": "± 55811",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45790304,
            "range": "± 389981",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20963921,
            "range": "± 724627",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204235940,
            "range": "± 6602925",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46478881,
            "range": "± 5397146",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218697798,
            "range": "± 9952302",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105476842,
            "range": "± 2189713",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45998895,
            "range": "± 411373",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20908179,
            "range": "± 337859",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7726938,
            "range": "± 118559",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4737326,
            "range": "± 37404",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4747856,
            "range": "± 20221",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578306,
            "range": "± 12437",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 325952,
            "range": "± 2916",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147661,
            "range": "± 1075",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 644504,
            "range": "± 12777",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 316441,
            "range": "± 9868",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1326192,
            "range": "± 19181",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 826905,
            "range": "± 31342",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2946344,
            "range": "± 30901",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2008882,
            "range": "± 42956",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6143415,
            "range": "± 85859",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3766052,
            "range": "± 107189",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 2030292386,
            "range": "± 51066335",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 662476788,
            "range": "± 12658617",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "932fdd511970b8e1a196aaedfa7b4936ae8a84c2",
          "message": "Create struct for column constants. (#556)",
          "timestamp": "2024-04-02T16:14:47+03:00",
          "tree_id": "fd4015133bace8fc8ff8be0d05addfd06a1b648d",
          "url": "https://github.com/starkware-libs/stwo/commit/932fdd511970b8e1a196aaedfa7b4936ae8a84c2"
        },
        "date": 1712064603381,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74258935,
            "range": "± 781500",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 200824812,
            "range": "± 7079253",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1979355,
            "range": "± 13217",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26393108,
            "range": "± 435362",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113718,
            "range": "± 673",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269891,
            "range": "± 1877",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 565612,
            "range": "± 7208",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1191462,
            "range": "± 15125",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2475199,
            "range": "± 10508",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5205377,
            "range": "± 30212",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11244909,
            "range": "± 219517",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25746568,
            "range": "± 489028",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57498220,
            "range": "± 1141243",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121340998,
            "range": "± 1390578",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 257169664,
            "range": "± 2898655",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 550884490,
            "range": "± 7552725",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1185714024,
            "range": "± 7394049",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12363,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4466,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 306813,
            "range": "± 988",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3212579,
            "range": "± 26610",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46279859,
            "range": "± 708887",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20869602,
            "range": "± 114427",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203549057,
            "range": "± 1627894",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46321286,
            "range": "± 1127471",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215250158,
            "range": "± 12287951",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105148951,
            "range": "± 562341",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45764109,
            "range": "± 485956",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20869884,
            "range": "± 778776",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7732547,
            "range": "± 164245",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4739196,
            "range": "± 16979",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730203,
            "range": "± 11486",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576972,
            "range": "± 13352",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 630,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 323894,
            "range": "± 3706",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 153998,
            "range": "± 4474",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 636623,
            "range": "± 9513",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 306235,
            "range": "± 9749",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1339266,
            "range": "± 17947",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 823924,
            "range": "± 78151",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2897261,
            "range": "± 35791",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1882167,
            "range": "± 42987",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5793423,
            "range": "± 130020",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3473055,
            "range": "± 63249",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1902217203,
            "range": "± 16210113",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 651074435,
            "range": "± 9157671",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bc540de0387e78d7db9d4e6b43d926b321dcf277",
          "message": "Precompute alphas for quotients. (#558)",
          "timestamp": "2024-04-02T16:19:35+03:00",
          "tree_id": "f18c36aa5799933995fc969971ea47eff6f4e9ef",
          "url": "https://github.com/starkware-libs/stwo/commit/bc540de0387e78d7db9d4e6b43d926b321dcf277"
        },
        "date": 1712064822330,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 81092305,
            "range": "± 2590244",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 231521835,
            "range": "± 2484016",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2054013,
            "range": "± 37989",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26620979,
            "range": "± 403033",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113373,
            "range": "± 1249",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 270238,
            "range": "± 8744",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 566645,
            "range": "± 3386",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1195679,
            "range": "± 8663",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2574270,
            "range": "± 28038",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 6142588,
            "range": "± 61571",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13474217,
            "range": "± 227930",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 28285330,
            "range": "± 466242",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 62517650,
            "range": "± 923342",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 131054497,
            "range": "± 1693922",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 272187325,
            "range": "± 4077927",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 593599362,
            "range": "± 8282500",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1277490784,
            "range": "± 18751746",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12690,
            "range": "± 196",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4591,
            "range": "± 87",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 301347,
            "range": "± 3353",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3490019,
            "range": "± 31576",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45740500,
            "range": "± 250329",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21103378,
            "range": "± 481047",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205305347,
            "range": "± 4856606",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46174103,
            "range": "± 540830",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218040878,
            "range": "± 8839969",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105135725,
            "range": "± 1329910",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45671163,
            "range": "± 391546",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20849852,
            "range": "± 601917",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7755949,
            "range": "± 439229",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4741773,
            "range": "± 16144",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4739679,
            "range": "± 18019",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578254,
            "range": "± 12820",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 628,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 760,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322754,
            "range": "± 8127",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145574,
            "range": "± 1469",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 632346,
            "range": "± 18673",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310076,
            "range": "± 8465",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1322881,
            "range": "± 20345",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 913576,
            "range": "± 21533",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 3028747,
            "range": "± 25257",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1773206,
            "range": "± 30504",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6331936,
            "range": "± 132336",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4135792,
            "range": "± 102191",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1974610587,
            "range": "± 20252935",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 650569757,
            "range": "± 6516276",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddc8d3b12870abddea52c17edc9055101f8505f1",
          "message": "Batch inverse cpu quotient denominators. (#559)",
          "timestamp": "2024-04-02T16:24:27+03:00",
          "tree_id": "e4bfaf4fb7e5cc768a7c252c1c6fbd430d5df651",
          "url": "https://github.com/starkware-libs/stwo/commit/ddc8d3b12870abddea52c17edc9055101f8505f1"
        },
        "date": 1712065091623,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 71116104,
            "range": "± 1428891",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 164810257,
            "range": "± 11018318",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1962250,
            "range": "± 14227",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26474696,
            "range": "± 325653",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112870,
            "range": "± 402",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265526,
            "range": "± 1753",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 561874,
            "range": "± 3088",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1185879,
            "range": "± 6828",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2490410,
            "range": "± 46950",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5284220,
            "range": "± 17511",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11749154,
            "range": "± 120583",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26455097,
            "range": "± 455702",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 55984447,
            "range": "± 433888",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 120181025,
            "range": "± 1378538",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 246117031,
            "range": "± 2010325",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 536167523,
            "range": "± 4972542",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1216684288,
            "range": "± 12314251",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12260,
            "range": "± 123",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4484,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 292823,
            "range": "± 1738",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3180213,
            "range": "± 15445",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45526016,
            "range": "± 886162",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20750801,
            "range": "± 461866",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204151208,
            "range": "± 4186308",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46217346,
            "range": "± 405621",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216682592,
            "range": "± 12122392",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105372951,
            "range": "± 4673072",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45668449,
            "range": "± 406259",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20825446,
            "range": "± 267672",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7757685,
            "range": "± 145562",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733642,
            "range": "± 11933",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733826,
            "range": "± 13617",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 575939,
            "range": "± 11049",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 759,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319538,
            "range": "± 6496",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145603,
            "range": "± 887",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 628342,
            "range": "± 11363",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 310104,
            "range": "± 9245",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1287110,
            "range": "± 9560",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 695795,
            "range": "± 21644",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2854528,
            "range": "± 48930",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1270725,
            "range": "± 7047",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5092162,
            "range": "± 41801",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 2592258,
            "range": "± 28966",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 2245566931,
            "range": "± 16099291",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 333740734,
            "range": "± 1509516",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e2d213e467860d26a809f676ff593c3c1a06173a",
          "message": "Implement Backend for AVX. (#561)",
          "timestamp": "2024-04-02T16:29:38+03:00",
          "tree_id": "477cc8de02dd311098c56a171822a6b754255cf0",
          "url": "https://github.com/starkware-libs/stwo/commit/e2d213e467860d26a809f676ff593c3c1a06173a"
        },
        "date": 1712065409223,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 79700784,
            "range": "± 2550071",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 224847385,
            "range": "± 8561780",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1983618,
            "range": "± 23239",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26482187,
            "range": "± 267955",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 114620,
            "range": "± 1038",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269730,
            "range": "± 13335",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564310,
            "range": "± 4709",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1182740,
            "range": "± 6710",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2522731,
            "range": "± 22612",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5543470,
            "range": "± 88602",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12654604,
            "range": "± 280888",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 28054226,
            "range": "± 302954",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59661679,
            "range": "± 799638",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 128016454,
            "range": "± 1597959",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 261066670,
            "range": "± 2437448",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 556764442,
            "range": "± 9477099",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1253915908,
            "range": "± 21562647",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12334,
            "range": "± 122",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4498,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297297,
            "range": "± 3397",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3411929,
            "range": "± 80508",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45655200,
            "range": "± 271267",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20727827,
            "range": "± 347363",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203848711,
            "range": "± 3452159",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46646969,
            "range": "± 1163150",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217452299,
            "range": "± 8808996",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105537187,
            "range": "± 4823213",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45721477,
            "range": "± 812175",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20887167,
            "range": "± 477682",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7741600,
            "range": "± 318405",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734309,
            "range": "± 16032",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738016,
            "range": "± 15034",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577649,
            "range": "± 8355",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 327018,
            "range": "± 3506",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145929,
            "range": "± 1349",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 633122,
            "range": "± 17227",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 330431,
            "range": "± 14019",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1317895,
            "range": "± 13012",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 768191,
            "range": "± 20702",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2941515,
            "range": "± 33595",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1928147,
            "range": "± 56935",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6076606,
            "range": "± 124504",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3883013,
            "range": "± 110412",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1629629815,
            "range": "± 16932411",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 345279109,
            "range": "± 5082838",
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
          "id": "14fa2ace556fad4d3b1e5aad5bc3663b955e2812",
          "message": "Use simple merkle tree (#526)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/526)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T09:29:30+03:00",
          "tree_id": "10529374ee8ca02c29ad2329224c27497ce7d863",
          "url": "https://github.com/starkware-libs/stwo/commit/14fa2ace556fad4d3b1e5aad5bc3663b955e2812"
        },
        "date": 1712126586085,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77202230,
            "range": "± 970264",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 214099762,
            "range": "± 2438007",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1962089,
            "range": "± 12700",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26382900,
            "range": "± 235373",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113278,
            "range": "± 1020",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266294,
            "range": "± 3473",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 567752,
            "range": "± 13970",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1189382,
            "range": "± 17614",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2468259,
            "range": "± 19282",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5205333,
            "range": "± 53965",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12038076,
            "range": "± 365551",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26502630,
            "range": "± 504795",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59233714,
            "range": "± 439429",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121376016,
            "range": "± 808766",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 258531578,
            "range": "± 2568607",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 544728066,
            "range": "± 7058752",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1194406334,
            "range": "± 5707866",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12353,
            "range": "± 133",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4462,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 304159,
            "range": "± 3810",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3214538,
            "range": "± 28735",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45606498,
            "range": "± 934992",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20919162,
            "range": "± 258838",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203405909,
            "range": "± 1500010",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46000790,
            "range": "± 747102",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216060712,
            "range": "± 9532808",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104827160,
            "range": "± 654112",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45617174,
            "range": "± 721363",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20865507,
            "range": "± 551316",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7756867,
            "range": "± 86957",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734551,
            "range": "± 14763",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4736356,
            "range": "± 13993",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577239,
            "range": "± 11708",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321161,
            "range": "± 6759",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 147491,
            "range": "± 2642",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 618246,
            "range": "± 6658",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 303158,
            "range": "± 9641",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1313698,
            "range": "± 25889",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 820786,
            "range": "± 24393",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2901628,
            "range": "± 81169",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1890638,
            "range": "± 64303",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5807790,
            "range": "± 222154",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3525552,
            "range": "± 55312",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1583550811,
            "range": "± 10577661",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 340391223,
            "range": "± 5419720",
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
          "id": "9298cc93fe690f2f690770aebb6c738957e512e8",
          "message": "simple merkle benchmark (#527)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/527)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T09:33:42+03:00",
          "tree_id": "a476f512c0fdc75d3a278368b4e2a1b087134f37",
          "url": "https://github.com/starkware-libs/stwo/commit/9298cc93fe690f2f690770aebb6c738957e512e8"
        },
        "date": 1712126892560,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76594038,
            "range": "± 564024",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 171592005,
            "range": "± 11400009",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1965450,
            "range": "± 22722",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26433520,
            "range": "± 734514",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113181,
            "range": "± 1645",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266146,
            "range": "± 1928",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 560468,
            "range": "± 3826",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1179697,
            "range": "± 18518",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2461875,
            "range": "± 16263",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5167197,
            "range": "± 46359",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11022851,
            "range": "± 57517",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 24277058,
            "range": "± 225248",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 55610445,
            "range": "± 635632",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 120469090,
            "range": "± 1090336",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 247291275,
            "range": "± 1603491",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 526336522,
            "range": "± 5145525",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1161193813,
            "range": "± 21503645",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12350,
            "range": "± 91",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4471,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 324494,
            "range": "± 2186",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3143373,
            "range": "± 56280",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45909415,
            "range": "± 516141",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20826025,
            "range": "± 242734",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203052104,
            "range": "± 3986528",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46254935,
            "range": "± 2798234",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215404357,
            "range": "± 11809207",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105605125,
            "range": "± 3484297",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45770405,
            "range": "± 406995",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20761790,
            "range": "± 303349",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7703214,
            "range": "± 80213",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733361,
            "range": "± 14822",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728097,
            "range": "± 7564",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576743,
            "range": "± 6504",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 2571325471,
            "range": "± 21370299",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 318769,
            "range": "± 3021",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 145363,
            "range": "± 900",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 639138,
            "range": "± 76991",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 309206,
            "range": "± 3529",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1318412,
            "range": "± 25740",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 779133,
            "range": "± 49298",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2848545,
            "range": "± 38308",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1640273,
            "range": "± 18213",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5752167,
            "range": "± 111093",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3255892,
            "range": "± 64765",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1557455544,
            "range": "± 11175311",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 339023101,
            "range": "± 1337019",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e1e590be477af8a5a601fbeb4ebf7381cbe1aa1a",
          "message": "Generate secure powers helper function (#553)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/553)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T10:04:01+03:00",
          "tree_id": "d52924f18f28d79f0e586d18240fac938ab0e0c5",
          "url": "https://github.com/starkware-libs/stwo/commit/e1e590be477af8a5a601fbeb4ebf7381cbe1aa1a"
        },
        "date": 1712128722613,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 79273129,
            "range": "± 1072728",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 218158688,
            "range": "± 2809742",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1972552,
            "range": "± 24810",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26661839,
            "range": "± 582977",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113491,
            "range": "± 925",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266229,
            "range": "± 3775",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 560014,
            "range": "± 4567",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1180634,
            "range": "± 19292",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2472258,
            "range": "± 24282",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5276559,
            "range": "± 38069",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11696657,
            "range": "± 372246",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26972790,
            "range": "± 302073",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58980448,
            "range": "± 545805",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 122733068,
            "range": "± 1154301",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 258631008,
            "range": "± 1980140",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 551808701,
            "range": "± 6818412",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1234289583,
            "range": "± 15865121",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12318,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4490,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 292978,
            "range": "± 3392",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3225198,
            "range": "± 30733",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45641753,
            "range": "± 328715",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20813373,
            "range": "± 377057",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204146283,
            "range": "± 1295672",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46065284,
            "range": "± 652298",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1221874063,
            "range": "± 13485700",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104832854,
            "range": "± 1804538",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45751890,
            "range": "± 722885",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20791835,
            "range": "± 263513",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7731742,
            "range": "± 93942",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4737707,
            "range": "± 19428",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4731706,
            "range": "± 12320",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 576887,
            "range": "± 9036",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 2582702599,
            "range": "± 17232433",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319750,
            "range": "± 2781",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 149861,
            "range": "± 6744",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 622191,
            "range": "± 12705",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 301296,
            "range": "± 7999",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1295483,
            "range": "± 69778",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 721895,
            "range": "± 18456",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2900775,
            "range": "± 27648",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1911435,
            "range": "± 68404",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5834635,
            "range": "± 85523",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3462400,
            "range": "± 92096",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1599254901,
            "range": "± 19735578",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 342715293,
            "range": "± 2764675",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "29c0ee1ee702c32852b2175ec0ebc1c500804bfe",
          "message": "Constraint accumulator precompute random coeff powers (#554)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/554)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T10:07:55+03:00",
          "tree_id": "3f8543453abb6a80d455080b09df29d768d455bb",
          "url": "https://github.com/starkware-libs/stwo/commit/29c0ee1ee702c32852b2175ec0ebc1c500804bfe"
        },
        "date": 1712128945064,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75476413,
            "range": "± 862316",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 203491770,
            "range": "± 9730469",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1965113,
            "range": "± 144680",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26376515,
            "range": "± 244263",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113641,
            "range": "± 3874",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269300,
            "range": "± 1025",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563104,
            "range": "± 4494",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1186041,
            "range": "± 6064",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2467914,
            "range": "± 30625",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5180982,
            "range": "± 51085",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11253927,
            "range": "± 143496",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25006582,
            "range": "± 236921",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 56965331,
            "range": "± 581411",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 120770658,
            "range": "± 862028",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 254930695,
            "range": "± 2471068",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 535998902,
            "range": "± 4452886",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1245030024,
            "range": "± 12572470",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12304,
            "range": "± 207",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4538,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297899,
            "range": "± 1258",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3187570,
            "range": "± 19288",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45807144,
            "range": "± 1376261",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20809380,
            "range": "± 349466",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204444859,
            "range": "± 3586114",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46166014,
            "range": "± 344684",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1212973233,
            "range": "± 8486381",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105579484,
            "range": "± 1627464",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45715217,
            "range": "± 341685",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20903356,
            "range": "± 402021",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7719347,
            "range": "± 64649",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4731439,
            "range": "± 11057",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730003,
            "range": "± 17818",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578570,
            "range": "± 10683",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 98",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 2587585179,
            "range": "± 18379454",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 321825,
            "range": "± 8969",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148553,
            "range": "± 1728",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 633005,
            "range": "± 6640",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 298512,
            "range": "± 3843",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1312644,
            "range": "± 29976",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 782283,
            "range": "± 54952",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2878414,
            "range": "± 97254",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1270714,
            "range": "± 11374",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5814238,
            "range": "± 166200",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3290864,
            "range": "± 29496",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1561194555,
            "range": "± 13279612",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338835610,
            "range": "± 3999637",
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
          "id": "ea89ad5618ad4700672644e5e31f7579b05a11de",
          "message": "avx merkle (#528)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/528)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T14:13:00+03:00",
          "tree_id": "3b7f25f2bf8dea600aec967f673f73fd4a050bf9",
          "url": "https://github.com/starkware-libs/stwo/commit/ea89ad5618ad4700672644e5e31f7579b05a11de"
        },
        "date": 1712143700894,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 86773413,
            "range": "± 1168115",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 305635198,
            "range": "± 2497865",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2054201,
            "range": "± 49329",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26969242,
            "range": "± 648034",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 114333,
            "range": "± 1559",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267000,
            "range": "± 2914",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 570176,
            "range": "± 10727",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1185072,
            "range": "± 9921",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2508261,
            "range": "± 34850",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5495557,
            "range": "± 166458",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13562270,
            "range": "± 464004",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 30244476,
            "range": "± 784097",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 61568579,
            "range": "± 549861",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 128594097,
            "range": "± 2619334",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 266413671,
            "range": "± 2471581",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 638524798,
            "range": "± 11006473",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1272327653,
            "range": "± 16991900",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12489,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4553,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 300960,
            "range": "± 1939",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3591427,
            "range": "± 37099",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45718606,
            "range": "± 969506",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20974903,
            "range": "± 207386",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204338978,
            "range": "± 4888436",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 47739704,
            "range": "± 782159",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1240045102,
            "range": "± 18298825",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104964447,
            "range": "± 2002011",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45686700,
            "range": "± 295348",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20916254,
            "range": "± 792637",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7742907,
            "range": "± 126891",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4737828,
            "range": "± 12568",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4735994,
            "range": "± 7657",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 577051,
            "range": "± 11486",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 183236369,
            "range": "± 5563557",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 35201089,
            "range": "± 428234",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 322765,
            "range": "± 19920",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 148642,
            "range": "± 2122",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 637900,
            "range": "± 8399",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 313343,
            "range": "± 8632",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1319002,
            "range": "± 57291",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 720113,
            "range": "± 26026",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2954476,
            "range": "± 111344",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1841651,
            "range": "± 53491",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6442197,
            "range": "± 157613",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4092694,
            "range": "± 83449",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 2173321306,
            "range": "± 48078509",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 408317182,
            "range": "± 6177503",
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
          "id": "c3327b8884b56c84e8ae0c96e68a6da28bbdf745",
          "message": "Dumb down FRI (#529)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/529)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T14:19:46+03:00",
          "tree_id": "a5e0e068e315b37dde48819adf5499204e99b5ad",
          "url": "https://github.com/starkware-libs/stwo/commit/c3327b8884b56c84e8ae0c96e68a6da28bbdf745"
        },
        "date": 1712144080058,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 79760622,
            "range": "± 2070286",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 228162561,
            "range": "± 7522387",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1989946,
            "range": "± 19176",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26549545,
            "range": "± 257709",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113531,
            "range": "± 980",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269154,
            "range": "± 2007",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563367,
            "range": "± 2896",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1188036,
            "range": "± 6347",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2505144,
            "range": "± 22449",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5749676,
            "range": "± 243370",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13208273,
            "range": "± 81343",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 28717928,
            "range": "± 270902",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 60324990,
            "range": "± 2251894",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 129066289,
            "range": "± 2486509",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 269333090,
            "range": "± 5170339",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 575704324,
            "range": "± 9378682",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1265130087,
            "range": "± 18940916",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12449,
            "range": "± 200",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4494,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297807,
            "range": "± 1661",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3428862,
            "range": "± 36701",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45798073,
            "range": "± 614952",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20997006,
            "range": "± 506481",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203943861,
            "range": "± 4411289",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46091022,
            "range": "± 356844",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217842162,
            "range": "± 10312783",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105827219,
            "range": "± 1267365",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45706865,
            "range": "± 442386",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20785433,
            "range": "± 591923",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7729642,
            "range": "± 47159",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735538,
            "range": "± 12137",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4737694,
            "range": "± 23347",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 592425,
            "range": "± 13311",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 175481573,
            "range": "± 1442640",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30009263,
            "range": "± 280669",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 324546,
            "range": "± 2776",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 156253,
            "range": "± 4724",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 624844,
            "range": "± 6398",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 398737,
            "range": "± 33121",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1316116,
            "range": "± 10736",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 839236,
            "range": "± 19464",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 3005765,
            "range": "± 31601",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 2154680,
            "range": "± 70234",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 6277475,
            "range": "± 172562",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 4150164,
            "range": "± 32063",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1648873374,
            "range": "± 36108119",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 347758482,
            "range": "± 5473257",
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
          "id": "65b9bc059882bfba0bc8a10b871101f1642b2f85",
          "message": "FRI using simple merkle (#530)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/530)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T14:55:59+03:00",
          "tree_id": "ec7499bfabb05cdf6c9d9132aacc6dace5518db9",
          "url": "https://github.com/starkware-libs/stwo/commit/65b9bc059882bfba0bc8a10b871101f1642b2f85"
        },
        "date": 1712146231692,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74548084,
            "range": "± 778227",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 124863551,
            "range": "± 4851473",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1967255,
            "range": "± 26834",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26385521,
            "range": "± 504675",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112914,
            "range": "± 1473",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267361,
            "range": "± 2255",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 570867,
            "range": "± 3918",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1193350,
            "range": "± 9464",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2499457,
            "range": "± 21095",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5276850,
            "range": "± 28424",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10944465,
            "range": "± 45880",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 23208507,
            "range": "± 307842",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 53522498,
            "range": "± 477857",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 116904773,
            "range": "± 1078362",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 237141188,
            "range": "± 2588235",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 534877444,
            "range": "± 5732795",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1207185789,
            "range": "± 10661520",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12304,
            "range": "± 424",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4529,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 325127,
            "range": "± 1706",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3140253,
            "range": "± 17019",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45476879,
            "range": "± 577359",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20938020,
            "range": "± 775275",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203486989,
            "range": "± 2543659",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46047257,
            "range": "± 365910",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218090764,
            "range": "± 10696087",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105478749,
            "range": "± 2376973",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45585948,
            "range": "± 443863",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20784333,
            "range": "± 536045",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7715416,
            "range": "± 71378",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4732062,
            "range": "± 12682",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4756325,
            "range": "± 14382",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 591323,
            "range": "± 12649",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 752,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 171608803,
            "range": "± 1123255",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 24602200,
            "range": "± 681214",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/32768",
            "value": 319355,
            "range": "± 3766",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/32768",
            "value": 146498,
            "range": "± 1044",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/65536",
            "value": 622370,
            "range": "± 12612",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/65536",
            "value": 316306,
            "range": "± 7650",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/131072",
            "value": 1273405,
            "range": "± 9450",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/131072",
            "value": 684771,
            "range": "± 19181",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/262144",
            "value": 2841125,
            "range": "± 33244",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/262144",
            "value": 1588774,
            "range": "± 16347",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE2/524288",
            "value": 5661301,
            "range": "± 81352",
            "unit": "ns/iter"
          },
          {
            "name": "Comparison of hashing algorithms and caching overhead/BLAKE3/524288",
            "value": 3178260,
            "range": "± 14529",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1572739993,
            "range": "± 11348970",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 335017767,
            "range": "± 4822184",
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
          "id": "f3da0ed3fccf130007d87904362823129715775b",
          "message": "Remove old merkle (#531)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/531)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T18:19:29+03:00",
          "tree_id": "98326ffce07efdde0ec5131867587aae6330914b",
          "url": "https://github.com/starkware-libs/stwo/commit/f3da0ed3fccf130007d87904362823129715775b"
        },
        "date": 1712158411025,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77668713,
            "range": "± 3317803",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 215143950,
            "range": "± 5917784",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1976230,
            "range": "± 21165",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26447605,
            "range": "± 632387",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113211,
            "range": "± 1069",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265704,
            "range": "± 2417",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 572398,
            "range": "± 4296",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1198335,
            "range": "± 7049",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2481365,
            "range": "± 22838",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5223122,
            "range": "± 138492",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11965918,
            "range": "± 362549",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26727947,
            "range": "± 432040",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58636363,
            "range": "± 450085",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123861308,
            "range": "± 1933836",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 260566773,
            "range": "± 2438350",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 553570860,
            "range": "± 5563803",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1240081738,
            "range": "± 4762622",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12388,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4488,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 300429,
            "range": "± 2438",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3230279,
            "range": "± 32499",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45751480,
            "range": "± 185696",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20787285,
            "range": "± 435528",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204341244,
            "range": "± 3355105",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46370855,
            "range": "± 805069",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219332186,
            "range": "± 16487216",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104983474,
            "range": "± 2380648",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46027703,
            "range": "± 202303",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20764003,
            "range": "± 388330",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7763965,
            "range": "± 122673",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4739238,
            "range": "± 10652",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733005,
            "range": "± 10587",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 588959,
            "range": "± 8280",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170683871,
            "range": "± 2768797",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30010849,
            "range": "± 657546",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1698042729,
            "range": "± 59165823",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 347650695,
            "range": "± 4433995",
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
          "id": "7dedf53d40d2953567e6604432e153af02b5d38e",
          "message": "AVX quotients (#488)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/488)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-03T18:24:36+03:00",
          "tree_id": "875fdbb95d92d2fe9d41886e92bb39351f2e71b0",
          "url": "https://github.com/starkware-libs/stwo/commit/7dedf53d40d2953567e6604432e153af02b5d38e"
        },
        "date": 1712158618557,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 81516814,
            "range": "± 1110416",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 226695153,
            "range": "± 3302477",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1978565,
            "range": "± 36770",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26496979,
            "range": "± 697131",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113097,
            "range": "± 928",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269646,
            "range": "± 1487",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 560592,
            "range": "± 7514",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1179939,
            "range": "± 11944",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2482901,
            "range": "± 33070",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5841360,
            "range": "± 178833",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13234640,
            "range": "± 164946",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 29306356,
            "range": "± 156275",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 61373529,
            "range": "± 747875",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 127134281,
            "range": "± 1490589",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 262031237,
            "range": "± 1788550",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 581595574,
            "range": "± 4546582",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1221097098,
            "range": "± 18052196",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12386,
            "range": "± 197",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4481,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 298358,
            "range": "± 2826",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3380577,
            "range": "± 50112",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46223874,
            "range": "± 759216",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20874306,
            "range": "± 890566",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204095377,
            "range": "± 1792049",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46042648,
            "range": "± 487268",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218569798,
            "range": "± 11697838",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105141785,
            "range": "± 727451",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46107481,
            "range": "± 628283",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20736553,
            "range": "± 633082",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7756989,
            "range": "± 34066",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734285,
            "range": "± 31246",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4739169,
            "range": "± 16441",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590412,
            "range": "± 13852",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170109103,
            "range": "± 2862062",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30069009,
            "range": "± 147404",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1627514272,
            "range": "± 20804304",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 347315363,
            "range": "± 4774213",
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
          "id": "9d91b8a1cafe990d2b5cd75895bcca1731095276",
          "message": "Fri AVX ops (#563)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/563)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T12:11:11+03:00",
          "tree_id": "d6177141e9951e5c800d9d5760ac1cc9244b0970",
          "url": "https://github.com/starkware-libs/stwo/commit/9d91b8a1cafe990d2b5cd75895bcca1731095276"
        },
        "date": 1712222707474,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75969934,
            "range": "± 948198",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 199695092,
            "range": "± 10522955",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1964519,
            "range": "± 25177",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26372270,
            "range": "± 450595",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113293,
            "range": "± 932",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266108,
            "range": "± 2856",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 568136,
            "range": "± 8792",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1186546,
            "range": "± 10237",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2485583,
            "range": "± 12748",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5249629,
            "range": "± 32531",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11506564,
            "range": "± 158264",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25928008,
            "range": "± 438893",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57925493,
            "range": "± 298613",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123374409,
            "range": "± 1099971",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 257272302,
            "range": "± 2252047",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 554023939,
            "range": "± 6412803",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1242320171,
            "range": "± 23282174",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12356,
            "range": "± 108",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4459,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 299650,
            "range": "± 2398",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3276918,
            "range": "± 38327",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45994663,
            "range": "± 391967",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20670408,
            "range": "± 281500",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203145823,
            "range": "± 4128889",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46633928,
            "range": "± 2213397",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214227958,
            "range": "± 14375608",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105061020,
            "range": "± 1989525",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46213541,
            "range": "± 343227",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20701309,
            "range": "± 487944",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7721758,
            "range": "± 93874",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736945,
            "range": "± 27447",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730835,
            "range": "± 9122",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589975,
            "range": "± 12283",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 169335834,
            "range": "± 1220196",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29808375,
            "range": "± 211599",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1581177749,
            "range": "± 19209493",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338742953,
            "range": "± 2918169",
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
          "id": "8046f4f800bc44f4f85fd09d90b61b20c30389d0",
          "message": "CommitmentSchemeProver generic in backend (#489)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/489)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T14:05:14+03:00",
          "tree_id": "224e97d3370fce751b1db0af01e226d5e46fe326",
          "url": "https://github.com/starkware-libs/stwo/commit/8046f4f800bc44f4f85fd09d90b61b20c30389d0"
        },
        "date": 1712229530177,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77389936,
            "range": "± 2026995",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 224332867,
            "range": "± 5899365",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1969088,
            "range": "± 36206",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26504387,
            "range": "± 810309",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113701,
            "range": "± 1036",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265984,
            "range": "± 1591",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 566668,
            "range": "± 5486",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1184740,
            "range": "± 9583",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2472758,
            "range": "± 28956",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5159343,
            "range": "± 39981",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11882710,
            "range": "± 263317",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26353482,
            "range": "± 303597",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58753987,
            "range": "± 397357",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123410281,
            "range": "± 1296116",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 261583571,
            "range": "± 1953647",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 560015226,
            "range": "± 4257854",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1227985774,
            "range": "± 13546883",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12321,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4567,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 322358,
            "range": "± 4732",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3389071,
            "range": "± 40214",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45951530,
            "range": "± 345645",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20796695,
            "range": "± 478060",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203646217,
            "range": "± 4729950",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46192576,
            "range": "± 540997",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219147203,
            "range": "± 12425293",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105169946,
            "range": "± 5282272",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46047066,
            "range": "± 267319",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20828331,
            "range": "± 520506",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7758979,
            "range": "± 125625",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736457,
            "range": "± 16570",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738358,
            "range": "± 6963",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590683,
            "range": "± 15745",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170707257,
            "range": "± 845814",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29522603,
            "range": "± 261339",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1595670524,
            "range": "± 19121590",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 341140371,
            "range": "± 5813179",
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
          "id": "42e6cf6921bf909e215f3ffdf99b2aea126ee2c9",
          "message": "Make prove() generic in Backend (#490)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/490)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T14:08:39+03:00",
          "tree_id": "fceca053f4bf102b57734c38017e9e6f3512d9d5",
          "url": "https://github.com/starkware-libs/stwo/commit/42e6cf6921bf909e215f3ffdf99b2aea126ee2c9"
        },
        "date": 1712229650868,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75442173,
            "range": "± 1440999",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 211167218,
            "range": "± 2503979",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1965416,
            "range": "± 34833",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26379196,
            "range": "± 153449",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113228,
            "range": "± 1292",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 270158,
            "range": "± 2142",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 562900,
            "range": "± 4218",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1179152,
            "range": "± 17397",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2475783,
            "range": "± 13326",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5297497,
            "range": "± 32367",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11753624,
            "range": "± 220177",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27970308,
            "range": "± 392849",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59130662,
            "range": "± 310175",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123767353,
            "range": "± 911791",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 255870436,
            "range": "± 2429147",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 561086983,
            "range": "± 6129804",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1225723015,
            "range": "± 10871928",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12371,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4456,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 299900,
            "range": "± 2381",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3279869,
            "range": "± 26938",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45903438,
            "range": "± 549399",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20893848,
            "range": "± 250982",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203036674,
            "range": "± 965454",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46187684,
            "range": "± 1012241",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213446901,
            "range": "± 9198205",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104739203,
            "range": "± 1734132",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46011517,
            "range": "± 957087",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20754658,
            "range": "± 270992",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7761714,
            "range": "± 78359",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733173,
            "range": "± 5835",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4732808,
            "range": "± 21148",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 591325,
            "range": "± 7603",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 168446376,
            "range": "± 3822110",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28768128,
            "range": "± 133503",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1574582647,
            "range": "± 13679929",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 339915056,
            "range": "± 2399979",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "950abfef9359c60821f23a0c0081d24be41f4a7d",
          "message": "Component mask_points api (#567)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/567)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T15:16:28+03:00",
          "tree_id": "478ba8c9234dcb8c77384aefc9e0c6819f99c230",
          "url": "https://github.com/starkware-libs/stwo/commit/950abfef9359c60821f23a0c0081d24be41f4a7d"
        },
        "date": 1712233784095,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77264213,
            "range": "± 560702",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 217552577,
            "range": "± 4126514",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1972591,
            "range": "± 8215",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26540773,
            "range": "± 390118",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113437,
            "range": "± 1076",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269825,
            "range": "± 2375",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 567427,
            "range": "± 10212",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1184412,
            "range": "± 9785",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2481295,
            "range": "± 23754",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5311801,
            "range": "± 51648",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12008652,
            "range": "± 126158",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26513417,
            "range": "± 408871",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57869932,
            "range": "± 410965",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 122334097,
            "range": "± 1026871",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 254439446,
            "range": "± 2073961",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 550566096,
            "range": "± 4153106",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1233588788,
            "range": "± 12844368",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12334,
            "range": "± 61",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4462,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 298429,
            "range": "± 2730",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3292610,
            "range": "± 32468",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46296551,
            "range": "± 1409090",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20776956,
            "range": "± 210123",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203377313,
            "range": "± 1302318",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46106254,
            "range": "± 1135707",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217812473,
            "range": "± 12680374",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105497434,
            "range": "± 1632081",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46051977,
            "range": "± 712789",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21029470,
            "range": "± 562438",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7758693,
            "range": "± 84091",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4731919,
            "range": "± 18819",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4743803,
            "range": "± 12491",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590570,
            "range": "± 5611",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 169918447,
            "range": "± 2277832",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 31006918,
            "range": "± 196682",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1581248498,
            "range": "± 17211940",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 341752258,
            "range": "± 4356178",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "90e675a0a78e8e3d587449aae5ade041ff289954",
          "message": "Component remove mask API (#568)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/568)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T15:20:33+03:00",
          "tree_id": "f8f02a84d9f3a808b86aef0fa00e729e530c3566",
          "url": "https://github.com/starkware-libs/stwo/commit/90e675a0a78e8e3d587449aae5ade041ff289954"
        },
        "date": 1712234041528,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73491945,
            "range": "± 1021783",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 207518960,
            "range": "± 5659424",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1960741,
            "range": "± 46101",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26424521,
            "range": "± 1091017",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113787,
            "range": "± 771",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265899,
            "range": "± 885",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 559836,
            "range": "± 5652",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1179289,
            "range": "± 5559",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2479899,
            "range": "± 22222",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5206689,
            "range": "± 18801",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11332429,
            "range": "± 164308",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25579986,
            "range": "± 301929",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58205105,
            "range": "± 334083",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121315146,
            "range": "± 741525",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 252381320,
            "range": "± 1662738",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 535694157,
            "range": "± 3129495",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1200874683,
            "range": "± 10885617",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12348,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4483,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 291063,
            "range": "± 2079",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3179406,
            "range": "± 20568",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45971345,
            "range": "± 179138",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20741342,
            "range": "± 338484",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204968319,
            "range": "± 4371682",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46328459,
            "range": "± 459191",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216126998,
            "range": "± 11762792",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105455664,
            "range": "± 3703042",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45877931,
            "range": "± 327533",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20928977,
            "range": "± 344696",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7771277,
            "range": "± 134463",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4727514,
            "range": "± 10100",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729633,
            "range": "± 6158",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589434,
            "range": "± 9947",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 168493048,
            "range": "± 1881754",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30690512,
            "range": "± 104842",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1564357223,
            "range": "± 15497394",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 337325943,
            "range": "± 2501728",
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
          "id": "58a34520fdbc6bbbf08c3557416ba8aadadaf54a",
          "message": "WideFib test with AVX Backend (#492)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/492)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-04T16:24:46+03:00",
          "tree_id": "4c110ee452165a3f3881dcefe2e282219335d2d0",
          "url": "https://github.com/starkware-libs/stwo/commit/58a34520fdbc6bbbf08c3557416ba8aadadaf54a"
        },
        "date": 1712237935772,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 81420531,
            "range": "± 1282506",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 228023283,
            "range": "± 2224720",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1980758,
            "range": "± 16741",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26485195,
            "range": "± 357671",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113248,
            "range": "± 909",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269181,
            "range": "± 2497",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 569061,
            "range": "± 7895",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1196931,
            "range": "± 7069",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2500618,
            "range": "± 12508",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5456972,
            "range": "± 80739",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12967624,
            "range": "± 98271",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27279893,
            "range": "± 343313",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 60868541,
            "range": "± 723628",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 126060630,
            "range": "± 1018156",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 265300659,
            "range": "± 2806424",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 568817839,
            "range": "± 5225008",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1260778774,
            "range": "± 8461175",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12369,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4507,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 294982,
            "range": "± 2878",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3511531,
            "range": "± 21222",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45966098,
            "range": "± 495058",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20805141,
            "range": "± 355446",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205378062,
            "range": "± 3347412",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46084982,
            "range": "± 1805357",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1220500128,
            "range": "± 13002779",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105293281,
            "range": "± 2293909",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46296833,
            "range": "± 501050",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20871571,
            "range": "± 467322",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7761884,
            "range": "± 69324",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736823,
            "range": "± 19854",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4736803,
            "range": "± 11102",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590559,
            "range": "± 6160",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 171959565,
            "range": "± 1601489",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 31865345,
            "range": "± 204807",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1644914064,
            "range": "± 16918015",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 346154155,
            "range": "± 2078840",
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
          "id": "d75d283ce5d16390673ac935fb1e828c795edada",
          "message": "Better inverse (#560)",
          "timestamp": "2024-04-08T14:03:15+03:00",
          "tree_id": "b2a1ba787d91dae0f6c5646f84d47c19403912aa",
          "url": "https://github.com/starkware-libs/stwo/commit/d75d283ce5d16390673ac935fb1e828c795edada"
        },
        "date": 1712575032015,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75903734,
            "range": "± 825966",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 213959885,
            "range": "± 3490155",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1960242,
            "range": "± 16406",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26391764,
            "range": "± 282196",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113535,
            "range": "± 1075",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265661,
            "range": "± 3176",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564686,
            "range": "± 27344",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1185108,
            "range": "± 10952",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2470329,
            "range": "± 17675",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5245226,
            "range": "± 66288",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12016747,
            "range": "± 273670",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27435819,
            "range": "± 548979",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59026873,
            "range": "± 615193",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123689832,
            "range": "± 1526500",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 263703464,
            "range": "± 2773845",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 557672957,
            "range": "± 4629111",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1233586480,
            "range": "± 7925902",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12351,
            "range": "± 64",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4474,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297403,
            "range": "± 3230",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3258925,
            "range": "± 24490",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46001843,
            "range": "± 498147",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20784874,
            "range": "± 207553",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203790616,
            "range": "± 4617932",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46570430,
            "range": "± 968349",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218817455,
            "range": "± 10934201",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104444107,
            "range": "± 1391299",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45954406,
            "range": "± 879034",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20808210,
            "range": "± 292386",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7749383,
            "range": "± 91017",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4739662,
            "range": "± 16554",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4737623,
            "range": "± 50372",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589469,
            "range": "± 14362",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170909333,
            "range": "± 3266368",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29526489,
            "range": "± 350774",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1602037883,
            "range": "± 13207092",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 341219179,
            "range": "± 2685918",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f90d64d8077132afede9f03c5838fa92a95a3b08",
          "message": "Remove Mask struct (#569)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/569)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-15T13:46:04+03:00",
          "tree_id": "edc5bfd442210cfbd49869656754f8833a3b3adf",
          "url": "https://github.com/starkware-libs/stwo/commit/f90d64d8077132afede9f03c5838fa92a95a3b08"
        },
        "date": 1713178767468,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75713574,
            "range": "± 1502267",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 214079181,
            "range": "± 4678175",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1980510,
            "range": "± 20760",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26542397,
            "range": "± 417496",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113776,
            "range": "± 545",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265701,
            "range": "± 1603",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564515,
            "range": "± 7872",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1177867,
            "range": "± 12431",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2489083,
            "range": "± 32370",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5387803,
            "range": "± 86360",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12447217,
            "range": "± 142751",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26597441,
            "range": "± 283015",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59967032,
            "range": "± 504371",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 125680298,
            "range": "± 802987",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 269308885,
            "range": "± 4681928",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 552107292,
            "range": "± 7093976",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1259995767,
            "range": "± 11444164",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12504,
            "range": "± 100",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4490,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297562,
            "range": "± 3352",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3417411,
            "range": "± 39127",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46020197,
            "range": "± 648436",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20803230,
            "range": "± 127183",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205680347,
            "range": "± 4374542",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46141966,
            "range": "± 962024",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217362152,
            "range": "± 12386249",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105190739,
            "range": "± 782081",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45983189,
            "range": "± 664646",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21002851,
            "range": "± 219853",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7758808,
            "range": "± 50443",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733555,
            "range": "± 13776",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4739050,
            "range": "± 12566",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590816,
            "range": "± 23959",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170348275,
            "range": "± 1995930",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28947908,
            "range": "± 221252",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1608782410,
            "range": "± 19512051",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 342340255,
            "range": "± 3979642",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c8552c4fd54f4ce22cbe2aecfe7487fd43fec8a7",
          "message": "Update wide fibonacci to 256 columns (#575)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/575)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-15T13:50:33+03:00",
          "tree_id": "9e240f4d726ff1fd8b636884dbff8c04acef42bd",
          "url": "https://github.com/starkware-libs/stwo/commit/c8552c4fd54f4ce22cbe2aecfe7487fd43fec8a7"
        },
        "date": 1713178968792,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73001728,
            "range": "± 1202086",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 207277175,
            "range": "± 5201270",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1959981,
            "range": "± 9518",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26524052,
            "range": "± 176136",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113916,
            "range": "± 859",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 268409,
            "range": "± 2676",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564942,
            "range": "± 9452",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1185741,
            "range": "± 5985",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2493414,
            "range": "± 17744",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5344637,
            "range": "± 74116",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12559361,
            "range": "± 174495",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26961293,
            "range": "± 251774",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59140995,
            "range": "± 439648",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123931418,
            "range": "± 1179650",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 259403183,
            "range": "± 1314627",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 558624490,
            "range": "± 4503916",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1241684797,
            "range": "± 8962442",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12358,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4512,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 292955,
            "range": "± 3151",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3274032,
            "range": "± 43211",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46121817,
            "range": "± 937276",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20748248,
            "range": "± 205039",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 202910647,
            "range": "± 1415291",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46057147,
            "range": "± 1145343",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216127574,
            "range": "± 12058420",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105329526,
            "range": "± 1255002",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46072446,
            "range": "± 855534",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20909211,
            "range": "± 185905",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7750793,
            "range": "± 63604",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735012,
            "range": "± 16466",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4744460,
            "range": "± 23115",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590812,
            "range": "± 5649",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 171117257,
            "range": "± 2938558",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28588098,
            "range": "± 250337",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1665921555,
            "range": "± 35079234",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 346446176,
            "range": "± 2891083",
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
          "id": "a8d128b117d70ddc9b843d00a98da224402f8904",
          "message": "Add measurement logs (#493)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/493)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-15T14:31:35+03:00",
          "tree_id": "fb30f1517c51c5a39c1025cc9d9a092f185f24c6",
          "url": "https://github.com/starkware-libs/stwo/commit/a8d128b117d70ddc9b843d00a98da224402f8904"
        },
        "date": 1713181538125,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 72583320,
            "range": "± 641012",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 120481331,
            "range": "± 3179939",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1950660,
            "range": "± 11128",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26309809,
            "range": "± 328037",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112989,
            "range": "± 950",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 263797,
            "range": "± 3420",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564328,
            "range": "± 31438",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1178512,
            "range": "± 9978",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2465523,
            "range": "± 11158",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5207658,
            "range": "± 31868",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10905328,
            "range": "± 104330",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 22796537,
            "range": "± 82880",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 52346990,
            "range": "± 417339",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 116529241,
            "range": "± 847280",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 243543750,
            "range": "± 1147838",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 533092683,
            "range": "± 1892234",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1192720532,
            "range": "± 10749732",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12255,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4451,
            "range": "± 76",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297736,
            "range": "± 2311",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3198581,
            "range": "± 26657",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46101945,
            "range": "± 346891",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20643307,
            "range": "± 76261",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203636464,
            "range": "± 4393827",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46218205,
            "range": "± 1094916",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215850766,
            "range": "± 19969071",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104807646,
            "range": "± 1816700",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45932527,
            "range": "± 868102",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20709137,
            "range": "± 94344",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7700078,
            "range": "± 87491",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4719732,
            "range": "± 11167",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729103,
            "range": "± 6918",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 591038,
            "range": "± 10135",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 169186560,
            "range": "± 3985363",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 24491347,
            "range": "± 187682",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1570488115,
            "range": "± 15248068",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 334552114,
            "range": "± 6022293",
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
          "id": "ae2aa27aabcbd84490b35ea851e39195eee32b13",
          "message": "Use precomputed twiddles (#494)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/494)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-15T14:35:37+03:00",
          "tree_id": "d5cb090c4d1821d3b57c7b3603c5879f17b7dd28",
          "url": "https://github.com/starkware-libs/stwo/commit/ae2aa27aabcbd84490b35ea851e39195eee32b13"
        },
        "date": 1713181690455,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 68742113,
            "range": "± 979215",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 199968066,
            "range": "± 4308978",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1961035,
            "range": "± 9787",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26447970,
            "range": "± 214246",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113625,
            "range": "± 574",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 268834,
            "range": "± 2695",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 562869,
            "range": "± 5345",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1189979,
            "range": "± 9672",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2484811,
            "range": "± 37856",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5144559,
            "range": "± 48376",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11395060,
            "range": "± 247231",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26410839,
            "range": "± 369671",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58617675,
            "range": "± 587800",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121568913,
            "range": "± 1253966",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 244677103,
            "range": "± 1933609",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 545574682,
            "range": "± 7627380",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1220775547,
            "range": "± 7508913",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12370,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4547,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 298865,
            "range": "± 2706",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3220826,
            "range": "± 29435",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45895825,
            "range": "± 1150300",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21069880,
            "range": "± 172423",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205149428,
            "range": "± 1847040",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46405954,
            "range": "± 861992",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1216216141,
            "range": "± 10174196",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104991163,
            "range": "± 1319685",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46094820,
            "range": "± 724698",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20755199,
            "range": "± 178419",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7733782,
            "range": "± 32682",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4733892,
            "range": "± 14815",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4735402,
            "range": "± 20205",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 592086,
            "range": "± 7244",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 172821185,
            "range": "± 2527617",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29806042,
            "range": "± 118968",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1580799081,
            "range": "± 13859132",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 340045443,
            "range": "± 6220361",
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
          "id": "f988f99065cb2e6aaadd1080fc7aceefd00d8be1",
          "message": "Reuse commitment evaluation (#495)",
          "timestamp": "2024-04-15T17:17:58+03:00",
          "tree_id": "310758d82b268cf2121c483ccbf8209b0ac2e30d",
          "url": "https://github.com/starkware-libs/stwo/commit/f988f99065cb2e6aaadd1080fc7aceefd00d8be1"
        },
        "date": 1713191429715,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76880920,
            "range": "± 688796",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 169301699,
            "range": "± 7995678",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1959816,
            "range": "± 28183",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26586043,
            "range": "± 276276",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113033,
            "range": "± 1276",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267689,
            "range": "± 4868",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 560425,
            "range": "± 4156",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1180991,
            "range": "± 6532",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2467713,
            "range": "± 3926",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5149365,
            "range": "± 49094",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10927787,
            "range": "± 55169",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 23837857,
            "range": "± 256892",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 54946827,
            "range": "± 566020",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 119408641,
            "range": "± 704673",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 250895779,
            "range": "± 2191041",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 535160052,
            "range": "± 4486404",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1224336152,
            "range": "± 9946231",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12273,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4445,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 298271,
            "range": "± 2896",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3165038,
            "range": "± 20953",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45853433,
            "range": "± 340252",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20813833,
            "range": "± 311764",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203513946,
            "range": "± 3088688",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46182753,
            "range": "± 982346",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213650525,
            "range": "± 16968624",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104932114,
            "range": "± 2447847",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46127075,
            "range": "± 1072352",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20787808,
            "range": "± 302067",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7747220,
            "range": "± 87149",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735049,
            "range": "± 13955",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733282,
            "range": "± 15200",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 591339,
            "range": "± 9226",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 173432645,
            "range": "± 2764482",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28910475,
            "range": "± 437991",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1572215879,
            "range": "± 16375248",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 340522805,
            "range": "± 2558077",
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
          "id": "249695d2a2bed64090f40dc66873d029e7de0c01",
          "message": "Evaluation oods once (#576)",
          "timestamp": "2024-04-16T14:15:33+03:00",
          "tree_id": "cd4af9d351c86c13c6174d72740196a91062de03",
          "url": "https://github.com/starkware-libs/stwo/commit/249695d2a2bed64090f40dc66873d029e7de0c01"
        },
        "date": 1713266993534,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78367442,
            "range": "± 760728",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 229561083,
            "range": "± 2771754",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1969410,
            "range": "± 31008",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26487869,
            "range": "± 282097",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113979,
            "range": "± 1358",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266118,
            "range": "± 3034",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 560355,
            "range": "± 8201",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1178922,
            "range": "± 9945",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2470282,
            "range": "± 13270",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5249506,
            "range": "± 99441",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11963923,
            "range": "± 191133",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26615511,
            "range": "± 242742",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 55261191,
            "range": "± 584048",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 123146621,
            "range": "± 1308971",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 249612942,
            "range": "± 1410871",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 557508067,
            "range": "± 8726093",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1235360782,
            "range": "± 13466683",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12385,
            "range": "± 130",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4467,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 334140,
            "range": "± 4090",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3273842,
            "range": "± 25913",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46198555,
            "range": "± 624038",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20758010,
            "range": "± 453137",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203686125,
            "range": "± 3631124",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46204105,
            "range": "± 1340102",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219821573,
            "range": "± 11199829",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105880317,
            "range": "± 2964008",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45916221,
            "range": "± 1561281",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20810125,
            "range": "± 443840",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7733697,
            "range": "± 83649",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4734338,
            "range": "± 17117",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4744390,
            "range": "± 35499",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 592694,
            "range": "± 9894",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 175444466,
            "range": "± 3107577",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30769363,
            "range": "± 216304",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1589703834,
            "range": "± 19909936",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 346710354,
            "range": "± 3908611",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "61963ecf4a94ed0597bd3b64ba0d96c94b3948a4",
          "message": "Add multilinear extension (#565)",
          "timestamp": "2024-04-17T00:34:54+12:00",
          "tree_id": "372e550cc53629c85552d71bbfeb46484a9ce779",
          "url": "https://github.com/starkware-libs/stwo/commit/61963ecf4a94ed0597bd3b64ba0d96c94b3948a4"
        },
        "date": 1713271693881,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76993389,
            "range": "± 1106427",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 208948231,
            "range": "± 7474336",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1969476,
            "range": "± 25638",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26357498,
            "range": "± 421940",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113745,
            "range": "± 2038",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266149,
            "range": "± 4633",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 561797,
            "range": "± 3062",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1190766,
            "range": "± 12879",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2483680,
            "range": "± 21018",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5373793,
            "range": "± 55474",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12135389,
            "range": "± 219732",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 26518638,
            "range": "± 308462",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58058161,
            "range": "± 747735",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 122335116,
            "range": "± 875721",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 256740664,
            "range": "± 2731578",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 553441215,
            "range": "± 7340101",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1221341206,
            "range": "± 9169546",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12289,
            "range": "± 75",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4467,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 321450,
            "range": "± 3463",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3231829,
            "range": "± 52784",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45970289,
            "range": "± 520430",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20755557,
            "range": "± 221027",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203211483,
            "range": "± 2982454",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46036891,
            "range": "± 863053",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1213033261,
            "range": "± 18691369",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104650455,
            "range": "± 2706099",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46046304,
            "range": "± 993246",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20771128,
            "range": "± 334636",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7750511,
            "range": "± 70207",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4736119,
            "range": "± 20402",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4747127,
            "range": "± 34975",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589867,
            "range": "± 9088",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 171282980,
            "range": "± 2050364",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29492233,
            "range": "± 107727",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1587855634,
            "range": "± 17269975",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 342106957,
            "range": "± 1248323",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a5e69d9fccec5af095667a89c05607399727f139",
          "message": "Multi crate infrastructure (#579)",
          "timestamp": "2024-04-16T16:23:55+03:00",
          "tree_id": "d7e528c14df786bcca8e06d6d0ee55d324dff344",
          "url": "https://github.com/starkware-libs/stwo/commit/a5e69d9fccec5af095667a89c05607399727f139"
        },
        "date": 1713274682065,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 88566611,
            "range": "± 2768135",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 312513862,
            "range": "± 9884947",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2014473,
            "range": "± 43395",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26552929,
            "range": "± 1073740",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112661,
            "range": "± 2033",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 268933,
            "range": "± 1397",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 569408,
            "range": "± 2388",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1198437,
            "range": "± 9898",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2524714,
            "range": "± 38291",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5696230,
            "range": "± 317929",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13088746,
            "range": "± 844771",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 29662702,
            "range": "± 579951",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 66264398,
            "range": "± 986440",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 130607669,
            "range": "± 2316969",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 263472437,
            "range": "± 2166189",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 594253668,
            "range": "± 4575134",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1300383604,
            "range": "± 16888411",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12384,
            "range": "± 388",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4530,
            "range": "± 51",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 301408,
            "range": "± 2742",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3581802,
            "range": "± 149603",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46156950,
            "range": "± 903551",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20871355,
            "range": "± 319870",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204871851,
            "range": "± 4858904",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46372919,
            "range": "± 614727",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1222613291,
            "range": "± 13893169",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105804015,
            "range": "± 2679439",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45930069,
            "range": "± 271962",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20841073,
            "range": "± 421718",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7750057,
            "range": "± 144410",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4738770,
            "range": "± 20799",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4743027,
            "range": "± 6944",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 591429,
            "range": "± 9723",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 632,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 759,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 179805188,
            "range": "± 2775441",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 34806799,
            "range": "± 600883",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 2191742750,
            "range": "± 34448919",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 408418385,
            "range": "± 6147449",
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
          "id": "51f227b58bf0491d09daa89c1ad49a0fb2e42e27",
          "message": "Fixes (#564)\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/564)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-17T10:22:31+03:00",
          "tree_id": "f80668b681d415a1122d2cf6cd1274cf1ef5ff09",
          "url": "https://github.com/starkware-libs/stwo/commit/51f227b58bf0491d09daa89c1ad49a0fb2e42e27"
        },
        "date": 1713339431215,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74591405,
            "range": "± 754131",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 183864665,
            "range": "± 7877564",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1959977,
            "range": "± 20088",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26445389,
            "range": "± 639799",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112687,
            "range": "± 1607",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267569,
            "range": "± 2008",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564614,
            "range": "± 4319",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1182622,
            "range": "± 8133",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2468573,
            "range": "± 16955",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5205768,
            "range": "± 69898",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11129930,
            "range": "± 93927",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 24603366,
            "range": "± 403871",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 54774295,
            "range": "± 429207",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 119135592,
            "range": "± 752540",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 254289633,
            "range": "± 2144421",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 547479760,
            "range": "± 6733126",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1210152980,
            "range": "± 17528799",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12450,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4453,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 294100,
            "range": "± 2333",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3222268,
            "range": "± 62827",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46208795,
            "range": "± 329415",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20734715,
            "range": "± 424961",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204235560,
            "range": "± 3877513",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46416159,
            "range": "± 1872725",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217540089,
            "range": "± 15348062",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105491832,
            "range": "± 1870585",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45952784,
            "range": "± 288543",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20913218,
            "range": "± 217865",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7739352,
            "range": "± 103115",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4745415,
            "range": "± 15143",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4739952,
            "range": "± 8523",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589528,
            "range": "± 8213",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170859455,
            "range": "± 2105675",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28295720,
            "range": "± 310312",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1580715282,
            "range": "± 18608834",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338205656,
            "range": "± 2296506",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2723806a0fbc56e102c45baa3b1221f0ac8e603a",
          "message": "Rename package name to stwo-prover (#584)\n\nRename prover package name to stwo-prover",
          "timestamp": "2024-04-21T20:35:31+03:00",
          "tree_id": "a9ddf0d02ccc7a6cdff93b878e1b6b4e44fb7735",
          "url": "https://github.com/starkware-libs/stwo/commit/2723806a0fbc56e102c45baa3b1221f0ac8e603a"
        },
        "date": 1713721786854,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73223327,
            "range": "± 958814",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 172396361,
            "range": "± 21663690",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1959978,
            "range": "± 9994",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26522124,
            "range": "± 225528",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112933,
            "range": "± 1099",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 264820,
            "range": "± 2211",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 562152,
            "range": "± 6354",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1177302,
            "range": "± 22264",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2483820,
            "range": "± 19636",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5263302,
            "range": "± 39132",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10795189,
            "range": "± 102738",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 23374742,
            "range": "± 217560",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 53739727,
            "range": "± 363886",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 118526223,
            "range": "± 1039549",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 247454731,
            "range": "± 1224679",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 543228776,
            "range": "± 4089603",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1211392590,
            "range": "± 11557814",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12352,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4522,
            "range": "± 156",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 290877,
            "range": "± 2599",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3151233,
            "range": "± 30804",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46090665,
            "range": "± 1666771",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20736044,
            "range": "± 167109",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203310904,
            "range": "± 3220068",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 45945221,
            "range": "± 976024",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1238368475,
            "range": "± 33656316",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105017986,
            "range": "± 1909052",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46007470,
            "range": "± 372977",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20794704,
            "range": "± 196019",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7733794,
            "range": "± 105008",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4739337,
            "range": "± 15426",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728320,
            "range": "± 12269",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589188,
            "range": "± 12170",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170273380,
            "range": "± 3654290",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 25805149,
            "range": "± 431488",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1559552549,
            "range": "± 12660623",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338289197,
            "range": "± 3891665",
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
          "id": "355f2e5df78c7585abd24e082f27e3f864923bf0",
          "message": "Seperate Component trait\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/580)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-24T10:30:39+03:00",
          "tree_id": "5b07bbbe0bfc4e18b3a2ff5c3934ed1747d5713a",
          "url": "https://github.com/starkware-libs/stwo/commit/355f2e5df78c7585abd24e082f27e3f864923bf0"
        },
        "date": 1713944696518,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 83172170,
            "range": "± 5216948",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 242372180,
            "range": "± 14586224",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2003588,
            "range": "± 35180",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26520865,
            "range": "± 569976",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112714,
            "range": "± 3159",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 271598,
            "range": "± 1355",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563245,
            "range": "± 5484",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1189094,
            "range": "± 11610",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2519690,
            "range": "± 27319",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5841136,
            "range": "± 247423",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13443513,
            "range": "± 649786",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27734504,
            "range": "± 1508048",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58607244,
            "range": "± 3141263",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 130103145,
            "range": "± 2984856",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 270967123,
            "range": "± 14553950",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 584411935,
            "range": "± 18368694",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1292667119,
            "range": "± 27384424",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12417,
            "range": "± 186",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4505,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 295397,
            "range": "± 2942",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3514825,
            "range": "± 47799",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45927911,
            "range": "± 658785",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20924999,
            "range": "± 2745269",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 205222323,
            "range": "± 1679497",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46151749,
            "range": "± 1030889",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218255165,
            "range": "± 12104676",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105188407,
            "range": "± 1947020",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45872797,
            "range": "± 508114",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20977682,
            "range": "± 786713",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7764049,
            "range": "± 42724",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4741121,
            "range": "± 20591",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4737118,
            "range": "± 22965",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 590631,
            "range": "± 13387",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 173945205,
            "range": "± 2403295",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 31180111,
            "range": "± 1264175",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1744512938,
            "range": "± 38872281",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 359364430,
            "range": "± 10533361",
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
          "id": "eca8057c5e8667a402cc27b4b79d3b1f6cc99e49",
          "message": "Move commitment scheme directories\n\n\n\n<!-- Reviewable:start -->\nThis change is [<img src=\"https://reviewable.io/review_button.svg\" height=\"34\" align=\"absmiddle\" alt=\"Reviewable\"/>](https://reviewable.io/reviews/starkware-libs/stwo/587)\n<!-- Reviewable:end -->",
          "timestamp": "2024-04-25T11:55:31+03:00",
          "tree_id": "a45191eeb50c058c5357b488ca30f45428188749",
          "url": "https://github.com/starkware-libs/stwo/commit/eca8057c5e8667a402cc27b4b79d3b1f6cc99e49"
        },
        "date": 1714036208790,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 74613443,
            "range": "± 1049227",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 211364336,
            "range": "± 5857313",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1973980,
            "range": "± 22879",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26469816,
            "range": "± 436013",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112130,
            "range": "± 2303",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269281,
            "range": "± 5272",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 567286,
            "range": "± 2386",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1184257,
            "range": "± 6989",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2503407,
            "range": "± 29926",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5230451,
            "range": "± 213281",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12849440,
            "range": "± 593556",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27017879,
            "range": "± 396096",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59608674,
            "range": "± 1054758",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 127290126,
            "range": "± 1747039",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 257274848,
            "range": "± 2757880",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 557836374,
            "range": "± 7788551",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1252688153,
            "range": "± 12716408",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12403,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4497,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 291972,
            "range": "± 2511",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3207689,
            "range": "± 32547",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45762399,
            "range": "± 315886",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21016254,
            "range": "± 347832",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203501706,
            "range": "± 3769661",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46294437,
            "range": "± 381152",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1222752379,
            "range": "± 10588097",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105145799,
            "range": "± 4937329",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46064090,
            "range": "± 191439",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20844715,
            "range": "± 504494",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7801015,
            "range": "± 179597",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4742054,
            "range": "± 14765",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4738344,
            "range": "± 13445",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 588308,
            "range": "± 9150",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170930089,
            "range": "± 1921774",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28485157,
            "range": "± 257908",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1611630582,
            "range": "± 16430375",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 349294862,
            "range": "± 2926893",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1f36d1c35d6510a16e227222d684bff03bae871b",
          "message": "Add sumcheck prover and verifier (#566)",
          "timestamp": "2024-04-26T02:23:09+12:00",
          "tree_id": "6472d5139b8d026dc9643cfb1c7b4e0db3f83eb9",
          "url": "https://github.com/starkware-libs/stwo/commit/1f36d1c35d6510a16e227222d684bff03bae871b"
        },
        "date": 1714055864691,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 70115026,
            "range": "± 344542",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 206056095,
            "range": "± 5232998",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1959634,
            "range": "± 17035",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26429891,
            "range": "± 533338",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112701,
            "range": "± 1147",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 270656,
            "range": "± 7809",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 564689,
            "range": "± 6487",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1176257,
            "range": "± 7192",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2465562,
            "range": "± 16598",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5155633,
            "range": "± 145354",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11444924,
            "range": "± 220698",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25410258,
            "range": "± 480499",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57490432,
            "range": "± 468761",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 118867928,
            "range": "± 1452647",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 252831715,
            "range": "± 2906087",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 541936094,
            "range": "± 2697891",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1232644816,
            "range": "± 14511262",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12331,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4525,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 328018,
            "range": "± 5113",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3268411,
            "range": "± 26596",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45919374,
            "range": "± 541005",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20830248,
            "range": "± 403923",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204041621,
            "range": "± 3520644",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46312437,
            "range": "± 1280419",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219042398,
            "range": "± 14074949",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105495527,
            "range": "± 2330644",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45894812,
            "range": "± 426244",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20734154,
            "range": "± 313045",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7760776,
            "range": "± 133526",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4743498,
            "range": "± 18551",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4734554,
            "range": "± 9693",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 588374,
            "range": "± 7409",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 170546629,
            "range": "± 2276254",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30184206,
            "range": "± 144654",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1565241574,
            "range": "± 10694768",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 341835703,
            "range": "± 1776829",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "91828241+alonh5@users.noreply.github.com",
            "name": "Alon Haramati",
            "username": "alonh5"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1b4554eb0967cca0b75a53cbe4fdfbd71ad5b608",
          "message": "Configure wide fib. (#562)",
          "timestamp": "2024-04-30T11:30:37+03:00",
          "tree_id": "c6b670a0653c8a27cd0abb85b4ef3dc794203376",
          "url": "https://github.com/starkware-libs/stwo/commit/1b4554eb0967cca0b75a53cbe4fdfbd71ad5b608"
        },
        "date": 1714466705592,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 73784038,
            "range": "± 798955",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 192972789,
            "range": "± 9874593",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1962944,
            "range": "± 40687",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26461321,
            "range": "± 724566",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112730,
            "range": "± 813",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269879,
            "range": "± 4467",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 559789,
            "range": "± 2704",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1180918,
            "range": "± 7521",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2471450,
            "range": "± 23090",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5168639,
            "range": "± 137714",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11207767,
            "range": "± 163839",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 24961551,
            "range": "± 239148",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 56792955,
            "range": "± 575012",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 122438663,
            "range": "± 1616234",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 250952233,
            "range": "± 1531254",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 544901467,
            "range": "± 4228927",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1177426378,
            "range": "± 10619754",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12355,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4476,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 309173,
            "range": "± 4210",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3233818,
            "range": "± 36366",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45792204,
            "range": "± 192707",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21010969,
            "range": "± 391590",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204247523,
            "range": "± 3004206",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46374489,
            "range": "± 633868",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1227329860,
            "range": "± 13831639",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104487257,
            "range": "± 3379315",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45989575,
            "range": "± 421314",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21019191,
            "range": "± 407319",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7764090,
            "range": "± 120275",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4748763,
            "range": "± 10484",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729033,
            "range": "± 11014",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589425,
            "range": "± 10085",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 169365441,
            "range": "± 2847868",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 27699697,
            "range": "± 100426",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1571984802,
            "range": "± 14071955",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338841449,
            "range": "± 4232440",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70577611+shaharsamocha7@users.noreply.github.com",
            "name": "shaharsamocha7",
            "username": "shaharsamocha7"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b6a9f6e017499efed76ace5354b54ade6a2e543",
          "message": "Remove redundant TODO (#590)\n\n\nRemove redundant TODO",
          "timestamp": "2024-05-01T09:09:36+03:00",
          "tree_id": "049fd96ce19ed98fa4e55806958f3d1dd669a9dc",
          "url": "https://github.com/starkware-libs/stwo/commit/3b6a9f6e017499efed76ace5354b54ade6a2e543"
        },
        "date": 1714544641373,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77389880,
            "range": "± 1345545",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 206373217,
            "range": "± 4878904",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1973729,
            "range": "± 24978",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26369050,
            "range": "± 454047",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112078,
            "range": "± 1729",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265864,
            "range": "± 8734",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563389,
            "range": "± 4248",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1184826,
            "range": "± 7683",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2472540,
            "range": "± 13125",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5198473,
            "range": "± 55519",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11413429,
            "range": "± 318232",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25658445,
            "range": "± 585564",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57708643,
            "range": "± 756805",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121673173,
            "range": "± 1012952",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 252222539,
            "range": "± 2238958",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 543066703,
            "range": "± 5034782",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1238355025,
            "range": "± 15610651",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12365,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4477,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 328680,
            "range": "± 5205",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3218761,
            "range": "± 65910",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45768998,
            "range": "± 399084",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20914782,
            "range": "± 441295",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203878397,
            "range": "± 3937671",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46112525,
            "range": "± 1492070",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215698896,
            "range": "± 13553219",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105073536,
            "range": "± 2956642",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45905775,
            "range": "± 326492",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21077031,
            "range": "± 285197",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7764361,
            "range": "± 107171",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4739534,
            "range": "± 19320",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4729294,
            "range": "± 12705",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 588074,
            "range": "± 9285",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 169829754,
            "range": "± 1896847",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30368106,
            "range": "± 213237",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1568678900,
            "range": "± 16058390",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338829684,
            "range": "± 1458816",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0533122224e57e5f41b4c4b9d125bb37cce72465",
          "message": "Optimize base field inverse implementation (#571)",
          "timestamp": "2024-05-02T02:33:20+12:00",
          "tree_id": "d51db2e09212498195357df242da242977ab6aac",
          "url": "https://github.com/starkware-libs/stwo/commit/0533122224e57e5f41b4c4b9d125bb37cce72465"
        },
        "date": 1714574794153,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 86497004,
            "range": "± 1339181",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 253620075,
            "range": "± 6192813",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2027736,
            "range": "± 80704",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26489291,
            "range": "± 248792",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112668,
            "range": "± 655",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266719,
            "range": "± 1143",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563602,
            "range": "± 4138",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1186831,
            "range": "± 11120",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2513948,
            "range": "± 42681",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 6006615,
            "range": "± 180793",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13614640,
            "range": "± 283793",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 28663169,
            "range": "± 353788",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 63570673,
            "range": "± 1103147",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 131291672,
            "range": "± 1962051",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 282367768,
            "range": "± 3458912",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 605739640,
            "range": "± 5669995",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1344778255,
            "range": "± 20819501",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12379,
            "range": "± 129",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4535,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 313334,
            "range": "± 3717",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3549716,
            "range": "± 59538",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46429363,
            "range": "± 685498",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21199205,
            "range": "± 412449",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204437068,
            "range": "± 5465762",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46255452,
            "range": "± 467967",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1223438525,
            "range": "± 10185168",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105671876,
            "range": "± 693972",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45788348,
            "range": "± 892661",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21344320,
            "range": "± 540419",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7787292,
            "range": "± 65573",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4749951,
            "range": "± 21177",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4733602,
            "range": "± 13458",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 581959,
            "range": "± 15821",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 623,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 759,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 174045114,
            "range": "± 2272831",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 32819301,
            "range": "± 290591",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1789943518,
            "range": "± 22460651",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 361170948,
            "range": "± 3729053",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89b4fcdb081184832a84f2a05185ff337bfc3b5f",
          "message": "Enable 'rng.gen()'ing field types (#591)",
          "timestamp": "2024-05-02T02:49:55+12:00",
          "tree_id": "ccb8e10d7fd743a623c11d187bd499f8a3e3a180",
          "url": "https://github.com/starkware-libs/stwo/commit/89b4fcdb081184832a84f2a05185ff337bfc3b5f"
        },
        "date": 1714575782803,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 79632504,
            "range": "± 969301",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 232940887,
            "range": "± 5535761",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2003686,
            "range": "± 53935",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 28520301,
            "range": "± 438562",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 114353,
            "range": "± 1076",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266846,
            "range": "± 2190",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 577328,
            "range": "± 6857",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1220171,
            "range": "± 16121",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2557196,
            "range": "± 26719",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5705311,
            "range": "± 126596",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13618272,
            "range": "± 367156",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 28071361,
            "range": "± 304742",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58930162,
            "range": "± 1389883",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 129090680,
            "range": "± 1970964",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 258117548,
            "range": "± 2995169",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 580246636,
            "range": "± 6839845",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1290111516,
            "range": "± 15860556",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12508,
            "range": "± 205",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4520,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 302415,
            "range": "± 4832",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3419378,
            "range": "± 57890",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 47141523,
            "range": "± 674208",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21697095,
            "range": "± 531810",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 210199975,
            "range": "± 2621152",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 47376536,
            "range": "± 520426",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1238392261,
            "range": "± 9233704",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 108814077,
            "range": "± 1612517",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 47332141,
            "range": "± 583414",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21649401,
            "range": "± 609325",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7858367,
            "range": "± 83630",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4776768,
            "range": "± 31513",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4794099,
            "range": "± 20291",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 589789,
            "range": "± 14504",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 644,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 764,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 177178469,
            "range": "± 2799037",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30291894,
            "range": "± 434965",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1673349746,
            "range": "± 30947131",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 357489370,
            "range": "± 5649171",
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
          "id": "c26de6d6c3c17b096ada5a642b7c6f32a0aef9da",
          "message": "removed useless test (#624)",
          "timestamp": "2024-05-09T13:41:48+03:00",
          "tree_id": "f7509d1462b1215153b8df6d4d88bb12914167d0",
          "url": "https://github.com/starkware-libs/stwo/commit/c26de6d6c3c17b096ada5a642b7c6f32a0aef9da"
        },
        "date": 1715252137996,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 77056916,
            "range": "± 1118537",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 227190781,
            "range": "± 2091046",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1968676,
            "range": "± 6250",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26518825,
            "range": "± 480533",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112065,
            "range": "± 888",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269104,
            "range": "± 3607",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 561564,
            "range": "± 9548",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1180999,
            "range": "± 5923",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2473137,
            "range": "± 18538",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5223025,
            "range": "± 65835",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11702336,
            "range": "± 214038",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25999010,
            "range": "± 337413",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 56127739,
            "range": "± 849706",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 120969294,
            "range": "± 1347903",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 250908579,
            "range": "± 1996316",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 539291384,
            "range": "± 5697504",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1164606041,
            "range": "± 34093009",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12312,
            "range": "± 134",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4499,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 302407,
            "range": "± 4132",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3248832,
            "range": "± 37188",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45992799,
            "range": "± 397362",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20690206,
            "range": "± 335032",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203211762,
            "range": "± 3362776",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46040174,
            "range": "± 730616",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1215340071,
            "range": "± 14294491",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104719870,
            "range": "± 2733702",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45844259,
            "range": "± 281005",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20749824,
            "range": "± 440490",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7745533,
            "range": "± 71182",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4731396,
            "range": "± 9726",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4728026,
            "range": "± 9796",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 580653,
            "range": "± 9996",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 753,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 173890753,
            "range": "± 3021371",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30638710,
            "range": "± 160779",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1576641367,
            "range": "± 14435477",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 343923677,
            "range": "± 1939728",
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
          "id": "e2307092499fbf9d6db612374a0107f24e2582f0",
          "message": "removed depracted math utils (#625)",
          "timestamp": "2024-05-15T13:30:09+03:00",
          "tree_id": "bb84491c038aa55f124163572cf8a2c6bf982377",
          "url": "https://github.com/starkware-libs/stwo/commit/e2307092499fbf9d6db612374a0107f24e2582f0"
        },
        "date": 1715776791841,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 68477617,
            "range": "± 313385",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 205757074,
            "range": "± 2310128",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1956314,
            "range": "± 14882",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26185195,
            "range": "± 136253",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112081,
            "range": "± 795",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 267732,
            "range": "± 1546",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 563997,
            "range": "± 4881",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1188057,
            "range": "± 23707",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2482277,
            "range": "± 8710",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5201445,
            "range": "± 31860",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 10858815,
            "range": "± 113364",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 24185641,
            "range": "± 307079",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 52976922,
            "range": "± 581598",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 109989966,
            "range": "± 609280",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 230489318,
            "range": "± 1792702",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 494215997,
            "range": "± 3245814",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1216053439,
            "range": "± 10452432",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12464,
            "range": "± 353",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4447,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 291099,
            "range": "± 2493",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3155789,
            "range": "± 19912",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45819969,
            "range": "± 393311",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20973855,
            "range": "± 584027",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 202485968,
            "range": "± 6273073",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 45847073,
            "range": "± 209490",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1214067103,
            "range": "± 12944452",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105132509,
            "range": "± 953628",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46115717,
            "range": "± 2077825",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 20689545,
            "range": "± 267999",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7692733,
            "range": "± 54896",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4728779,
            "range": "± 14575",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730013,
            "range": "± 20281",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578842,
            "range": "± 9192",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 62",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 171555189,
            "range": "± 1671190",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29133745,
            "range": "± 184201",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1560200245,
            "range": "± 7976132",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338325611,
            "range": "± 4738731",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d611960ffbdf89a2ba5e9d4a0d899892a0a7396",
          "message": "Add SIMD backend arithmetic (#592)",
          "timestamp": "2024-05-15T14:09:31+01:00",
          "tree_id": "43dd9941a05e74ecea9a79dd6cc44bdf7c73fd95",
          "url": "https://github.com/starkware-libs/stwo/commit/8d611960ffbdf89a2ba5e9d4a0d899892a0a7396"
        },
        "date": 1715779451778,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 80537558,
            "range": "± 1526917",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 227992995,
            "range": "± 4238520",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1980161,
            "range": "± 67095",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26519750,
            "range": "± 189701",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112029,
            "range": "± 676",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 266030,
            "range": "± 1858",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 561214,
            "range": "± 6972",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1193300,
            "range": "± 10294",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2515121,
            "range": "± 31298",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5501245,
            "range": "± 84604",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12450824,
            "range": "± 263837",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27370094,
            "range": "± 209094",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 58218093,
            "range": "± 1089470",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 126977258,
            "range": "± 660335",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 264146841,
            "range": "± 3209231",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 581733759,
            "range": "± 6648342",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1273669272,
            "range": "± 14735479",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12360,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4481,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 297170,
            "range": "± 3113",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3358911,
            "range": "± 44872",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45972824,
            "range": "± 398878",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21218536,
            "range": "± 383478",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204591960,
            "range": "± 5398917",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46332969,
            "range": "± 351680",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1217212332,
            "range": "± 18143981",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104959690,
            "range": "± 1343549",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45873751,
            "range": "± 322054",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21192871,
            "range": "± 382524",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7758425,
            "range": "± 462325",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4738830,
            "range": "± 11294",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4734136,
            "range": "± 12161",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 579112,
            "range": "± 15262",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 626,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 172993877,
            "range": "± 1919624",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30899769,
            "range": "± 280126",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1633007334,
            "range": "± 12347359",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 349134908,
            "range": "± 4040015",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8ed4d95cb12cf07a287095af43519334d87f5c1a",
          "message": "Implement FieldOps for SIMD backend (#593)",
          "timestamp": "2024-05-15T14:14:53+01:00",
          "tree_id": "d80b892729b5f8d525c86a70b0ea8feaedc30012",
          "url": "https://github.com/starkware-libs/stwo/commit/8ed4d95cb12cf07a287095af43519334d87f5c1a"
        },
        "date": 1715779674871,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 82382053,
            "range": "± 982522",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 230037793,
            "range": "± 5557937",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 2001229,
            "range": "± 35522",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26536804,
            "range": "± 953842",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 113200,
            "range": "± 978",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 270560,
            "range": "± 2323",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 567740,
            "range": "± 4475",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1193230,
            "range": "± 10591",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2515283,
            "range": "± 25754",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5764808,
            "range": "± 105926",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 13126356,
            "range": "± 183444",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27576966,
            "range": "± 241333",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 59447921,
            "range": "± 688101",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 128207514,
            "range": "± 1041228",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 266363079,
            "range": "± 2113384",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 588916935,
            "range": "± 5052938",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1287134901,
            "range": "± 8622207",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12392,
            "range": "± 133",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4447,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 304668,
            "range": "± 2509",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3431123,
            "range": "± 80622",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45842280,
            "range": "± 355588",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21237893,
            "range": "± 534711",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204105596,
            "range": "± 3992645",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46283979,
            "range": "± 1094384",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1225636976,
            "range": "± 8346933",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105351714,
            "range": "± 2823252",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46092251,
            "range": "± 446829",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21392308,
            "range": "± 369946",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7763213,
            "range": "± 93559",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4742227,
            "range": "± 10413",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4744466,
            "range": "± 7916",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 580416,
            "range": "± 11646",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 627,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 756,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 174857199,
            "range": "± 2211607",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 30355477,
            "range": "± 287298",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1674436989,
            "range": "± 15039783",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 348721564,
            "range": "± 3595325",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d13da83983edc90c0d85cd93df0454d4f5b7af6e",
          "message": "Add bit_reverse for SIMD backend (#600)",
          "timestamp": "2024-05-15T14:29:58+01:00",
          "tree_id": "ad4c778c4eda21dc3ea328682c1897d5ccaf2060",
          "url": "https://github.com/starkware-libs/stwo/commit/d13da83983edc90c0d85cd93df0454d4f5b7af6e"
        },
        "date": 1715780636210,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 78417465,
            "range": "± 1627667",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 221092101,
            "range": "± 4402764",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1978860,
            "range": "± 26183",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26565478,
            "range": "± 776314",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112746,
            "range": "± 1975",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 268926,
            "range": "± 1777",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 569256,
            "range": "± 3629",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1191398,
            "range": "± 10956",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2481809,
            "range": "± 29067",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5478246,
            "range": "± 64201",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 12626317,
            "range": "± 187502",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 27189325,
            "range": "± 251034",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57491501,
            "range": "± 480850",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 125095501,
            "range": "± 1392876",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 262983859,
            "range": "± 2947230",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 580092580,
            "range": "± 6270197",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1257170951,
            "range": "± 16556133",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12318,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4445,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 316916,
            "range": "± 3711",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3492762,
            "range": "± 44976",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 46136540,
            "range": "± 254358",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21264348,
            "range": "± 257647",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203730686,
            "range": "± 4128472",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46178465,
            "range": "± 1116515",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1232433676,
            "range": "± 15858267",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 104832943,
            "range": "± 2103151",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45784068,
            "range": "± 1153552",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21340807,
            "range": "± 689346",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7742034,
            "range": "± 74547",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4735939,
            "range": "± 15926",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4744837,
            "range": "± 82046",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 579234,
            "range": "± 7787",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 757,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 173982438,
            "range": "± 3010916",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 29601962,
            "range": "± 243302",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1639409716,
            "range": "± 16166467",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 346241700,
            "range": "± 2657056",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "24935cf252ff8f42883edbe3aa33a8e82c33ec8f",
          "message": "Add bit_reverse for SIMD backend (#594)",
          "timestamp": "2024-05-16T03:01:28+12:00",
          "tree_id": "8698410f7660ce002bec5fd9f3ee1d3217396929",
          "url": "https://github.com/starkware-libs/stwo/commit/24935cf252ff8f42883edbe3aa33a8e82c33ec8f"
        },
        "date": 1715786044008,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 75200492,
            "range": "± 984982",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 181067838,
            "range": "± 6812437",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1953839,
            "range": "± 17539",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26533348,
            "range": "± 406223",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 112226,
            "range": "± 1228",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 269382,
            "range": "± 4702",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 565126,
            "range": "± 8549",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1188328,
            "range": "± 7772",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2482236,
            "range": "± 26059",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5162738,
            "range": "± 48723",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11159135,
            "range": "± 87628",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 25203665,
            "range": "± 264789",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 57100584,
            "range": "± 502430",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 121614285,
            "range": "± 1117887",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 252604445,
            "range": "± 1488331",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 539280245,
            "range": "± 7111961",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1204532718,
            "range": "± 10025546",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12310,
            "range": "± 83",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4484,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 309750,
            "range": "± 4741",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3173812,
            "range": "± 17255",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45817616,
            "range": "± 730080",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21014384,
            "range": "± 126817",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 203960172,
            "range": "± 2447004",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46116485,
            "range": "± 702436",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1218833450,
            "range": "± 10294651",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105173903,
            "range": "± 1124951",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45705109,
            "range": "± 838263",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21080621,
            "range": "± 1584586",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7719670,
            "range": "± 25988",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4727586,
            "range": "± 10808",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4731846,
            "range": "± 13121",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 578933,
            "range": "± 14539",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 624,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 754,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 172495582,
            "range": "± 1948281",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 27973062,
            "range": "± 180471",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1565842774,
            "range": "± 11598472",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 339443699,
            "range": "± 6051937",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrewmilson@users.noreply.github.com",
            "name": "Andrew Milson",
            "username": "andrewmilson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f2c90cd025a645face7acf4c5c04ceaacde279e6",
          "message": "Add blake2s implementation for SIMD backend (#601)",
          "timestamp": "2024-05-16T03:16:07+12:00",
          "tree_id": "e06728e3c48e2e2258a3ee030e19f3443a115c23",
          "url": "https://github.com/starkware-libs/stwo/commit/f2c90cd025a645face7acf4c5c04ceaacde279e6"
        },
        "date": 1715786910625,
        "tool": "cargo",
        "benches": [
          {
            "name": "avx bit_rev 26bit",
            "value": 76928690,
            "range": "± 1449146",
            "unit": "ns/iter"
          },
          {
            "name": "cpu bit_rev 24bit",
            "value": 189547484,
            "range": "± 5545930",
            "unit": "ns/iter"
          },
          {
            "name": "avx eval_at_secure_field_point 2^20",
            "value": 1962706,
            "range": "± 16735",
            "unit": "ns/iter"
          },
          {
            "name": "cpu eval_at_secure_field_point 2^20",
            "value": 26312537,
            "range": "± 213076",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/16",
            "value": 111774,
            "range": "± 656",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/17",
            "value": 265461,
            "range": "± 1136",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/18",
            "value": 561797,
            "range": "± 2932",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/19",
            "value": 1192790,
            "range": "± 10169",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/20",
            "value": 2483642,
            "range": "± 15328",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/21",
            "value": 5124733,
            "range": "± 27933",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/22",
            "value": 11090203,
            "range": "± 70464",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/23",
            "value": 24458931,
            "range": "± 397252",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/24",
            "value": 55184793,
            "range": "± 401548",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/25",
            "value": 119434154,
            "range": "± 627483",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/26",
            "value": 253837300,
            "range": "± 2418648",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/27",
            "value": 539692890,
            "range": "± 6111660",
            "unit": "ns/iter"
          },
          {
            "name": "iffts/avx ifft/28",
            "value": 1230765459,
            "range": "± 26530506",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft_vecwise_loop 2^14",
            "value": 12346,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx ifft3_loop 2^14",
            "value": 4506,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "ifft parts/avx transpose_vecs 2^20",
            "value": 328949,
            "range": "± 2294",
            "unit": "ns/iter"
          },
          {
            "name": "avx rfft 20bit",
            "value": 3164893,
            "range": "± 21079",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45842504,
            "range": "± 308502",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21154778,
            "range": "± 408498",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 mul",
            "value": 204168619,
            "range": "± 3857673",
            "unit": "ns/iter"
          },
          {
            "name": "CM31 add",
            "value": 46086534,
            "range": "± 306139",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField mul",
            "value": 1219869593,
            "range": "± 13420864",
            "unit": "ns/iter"
          },
          {
            "name": "SecureField add",
            "value": 105193743,
            "range": "± 1894972",
            "unit": "ns/iter"
          },
          {
            "name": "M31 mul",
            "value": 45748231,
            "range": "± 410051",
            "unit": "ns/iter"
          },
          {
            "name": "M31 add",
            "value": 21121736,
            "range": "± 444187",
            "unit": "ns/iter"
          },
          {
            "name": "mul_avx512",
            "value": 7744794,
            "range": "± 113332",
            "unit": "ns/iter"
          },
          {
            "name": "add_avx512",
            "value": 4729779,
            "range": "± 9698",
            "unit": "ns/iter"
          },
          {
            "name": "sub_avx512",
            "value": 4730052,
            "range": "± 5114",
            "unit": "ns/iter"
          },
          {
            "name": "fold_line",
            "value": 579168,
            "range": "± 9572",
            "unit": "ns/iter"
          },
          {
            "name": "RowMajorMatrix M31 24x24 mul",
            "value": 625,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "QM31 RowMajorMatrix 6x6 mul",
            "value": 755,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/cpu merkle",
            "value": 172477559,
            "range": "± 1382504",
            "unit": "ns/iter"
          },
          {
            "name": "merkle throughput/avx merkle",
            "value": 28192620,
            "range": "± 352032",
            "unit": "ns/iter"
          },
          {
            "name": "avx quotients 2^8 x 2^20",
            "value": 1579496416,
            "range": "± 12290506",
            "unit": "ns/iter"
          },
          {
            "name": "cpu quotients 2^8 x 2^16",
            "value": 338929490,
            "range": "± 1567337",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}