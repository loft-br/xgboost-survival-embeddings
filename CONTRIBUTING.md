## How to contribute to XGBSE

We welcome contributions from everyone!


### Getting Help

If you found a bug or need a new feature, you can submit an [issue](https://github.com/loft-br/xgboost-survival-embeddings/issues).

## Getting started with delopment

To start contributing with code or documentation you first need to fork the project. To clone your fork in your machine:

```sh
git clone git@github.com:your-username/xgboost-survival-embeddings.git

git remote add upstream https://github.com/loft-br/xgboost-survival-embeddings.git

```

This will create a folder called `xgboost-survival-embeddings` and will connect to the upstream(main repo).

We recommend you to create a virtualenv before starts to work with the code. And be able to run all tests locally before start to write new code.

### Installing development version

To install the development version of the library you can run `make install-dev` on your fork folder.

It'll also install a pre-commit hook that will check flake8 and black on each commit, to keep the code style consistent.

### Check format and run tests

To run linting, formatting and tests you can use `make check`.

If every check pass, you should be ready to start contributing.

## Code and documentation contributions

### Code standard

We use flake8 and black to ensure an uniform code style in the whole library.

In order to check if your code follow our style, you can run:

`make flake8 black`

### Tests

We use [pytest](https://docs.pytest.org/en/latest/) for testing. You should always write tests for your new feature.

After your development or fix you can check the tests by running:
`make test`


Thanks! :heart:

Loft Data Science Team
