def test_dataset(test_df):

    """Test dataset quality and integrity."""
    # schema adherence
    column_list = ['id', 'comment_text']
    test_df.expect_table_columns_to_match_ordered_list(column_list=column_list)

    # missing values
    test_df.expect_column_values_to_not_be_null(column="id")
    test_df.expect_column_values_to_not_be_null(column="comment_text")
    # unique values
    test_df.expect_column_values_to_be_unique(column="id")
    # type adherence
    test_df.expect_column_values_to_be_of_type(column="comment_text", type_="str")

    # Expectation suite
    expectation_suite = test_df.get_expectation_suite(discard_failed_expectations=False)
    results = test_df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]
