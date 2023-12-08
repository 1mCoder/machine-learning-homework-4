def test_dataset(train_df):
    
    # schema adherence
    column_list = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_df.expect_table_columns_to_match_ordered_list(column_list=column_list)

    # missing values
    train_df.expect_column_values_to_not_be_null(column="id")
    train_df.expect_column_values_to_not_be_null(column="comment_text")
    train_df.expect_column_values_to_not_be_null(column="toxic")
    train_df.expect_column_values_to_not_be_null(column="severe_toxic")
    train_df.expect_column_values_to_not_be_null(column="obscene")
    train_df.expect_column_values_to_not_be_null(column="threat")
    train_df.expect_column_values_to_not_be_null(column="insult")
    train_df.expect_column_values_to_not_be_null(column="identity_hate")
    # unique values
    train_df.expect_column_values_to_be_unique(column="id")
    # type adherence
    train_df.expect_column_values_to_be_of_type(column="comment_text", type_="str")

    # Expectation suite
    expectation_suite = train_df.get_expectation_suite(discard_failed_expectations=False)
    results = train_df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]
