"""
Module containing tools for evaluation of the model.

The tools are 2 variants of the tools from converso, the first based on StructuredTools and the second based on FormTools.

The tools for evaluation:
- have their _run or _run_when_complete methods deactivated (they just return a string)
- have a get_random_payload method that returns a random payload for the tool, used to generate test cases
"""
