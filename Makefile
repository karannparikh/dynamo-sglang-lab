.PHONY: test check render

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

check: test
	PYTHONPATH=src python3 -m compileall -q src tests docker

render:
	PYTHONPATH=src python3 -m dynamo_routing_harness render --output .local/rendered
