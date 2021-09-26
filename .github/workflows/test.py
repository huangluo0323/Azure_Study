name: Huangluo Test

on:
  push:
    branches:
      - master
      - slave
jobs:
  test_qa:
    if: ${{ github.ref == 'refs/heads/slave' }}
    runs-on: ubuntu-latest
    # Steps represent a sequence of tasks that will be executed as part of the job
    steps:
      - run: echo "Test qa"

  deployment_test_prd:
    if: ${{ github.ref == 'refs/heads/master' }}
    runs-on: ubuntu-latest
      - run: echo "Test prd"
