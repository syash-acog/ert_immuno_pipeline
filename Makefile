build:
	@docker build -t immuno-pipeline .

run: build
	@docker run --rm --name immuno-container -v "$(shell pwd)/data/output:/app/data/output" immuno-pipeline

stop:
	@docker stop immuno-container || true

clean: stop
	@docker rmi immuno-pipeline || true