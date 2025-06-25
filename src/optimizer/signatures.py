import dspy


class NLSISignature(dspy.Signature):
    user_utterance = dspy.InputField(desc="<user_utterance>", prefix="Dialogue:\n")
    user_profile = dspy.InputField(
        desc="<applicable_standing_instructions>",
        prefix="Applicable Standing Instructions:\n",
    )
    answer = dspy.OutputField(
        desc="API calls to solve the user task", prefix="API Calls:\n"
    )


class NLSISignatureAPI(dspy.Signature):
    api_schema = dspy.InputField(desc="<api_schema>", prefix="Schema:\n")
    user_utterance = dspy.InputField(desc="<user_utterance>", prefix="Dialogue:\n")
    user_profile = dspy.InputField(
        desc="<applicable_standing_instructions>",
        prefix="Applicable Standing Instructions:\n",
    )
    answer = dspy.OutputField(
        desc="API calls to solve the user task", prefix="API Calls:\n"
    )


class NLSISignatureSimpleTag(dspy.Signature):
    user_utterance = dspy.InputField(desc="<user_utterance>", prefix="Dialogue:\n")
    user_profile = dspy.InputField(
        desc="<applicable_standing_instructions>",
        prefix="Applicable Standing Instructions:\n",
    )
    tagged_user_profile = dspy.InputField(
        desc="<tagged applicable standing instructions>",
        prefix="Tagged Standing Instructions:\n",
    )
    answer = dspy.OutputField(
        desc="API calls to solve the user task", prefix="API Calls:\n"
    )


class NLSISignatureCoTTag(dspy.Signature):
    user_utterance = dspy.InputField(desc="<user_utterance>", prefix="Dialogue:\n")
    user_profile = dspy.InputField(
        desc="<applicable_standing_instructions>",
        prefix="Applicable Standing Instructions:\n",
    )
    tagged_user_profile = dspy.OutputField(
        desc="Tagged standing instructions",
        prefix="Tagged Standing Instructions:\n",
    )
    answer = dspy.OutputField(
        desc="API calls to solve the user task",
        prefix="API Calls:\n",
    )
