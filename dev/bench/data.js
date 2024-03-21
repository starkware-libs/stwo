window.BENCHMARK_DATA = {
  "lastUpdate": 1711020954012,
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
      }
    ]
  }
}