##
# Project Title
#
# @file
# @version 0.1

test_command:= python -m unittest
test_src = src/tests/

test:
	$(test_command) $(test_src)*.py
# end
