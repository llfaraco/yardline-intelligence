# Tests

The public repository currently contains bootstrap tests only. The original local research suite must be migrated with the implementation.

Required scientific tests include:

- temporal feature generation never reads future weeks;
- player identity joins use canonical IDs;
- missing values remain missing unless an explicit imputation rule applies;
- target scoring matches the Half-PPR definition;
- train/selection/test seasons remain disjoint;
- metric calculations match independently calculated fixtures;
- simulation intervals and CRPS behave correctly on controlled examples.

A passing bootstrap test suite is not evidence that the scientific pipeline has been reproduced.
