help:
	@echo "Immunogenicity Risk Pipeline - Docker Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make build              Build the Docker image"
	@echo "  make run                Run with default test data"
	@echo "  make run-custom         Run with custom inputs (see examples below)"
	@echo "  make stop               Stop running container"
	@echo "  make clean              Remove container and image"
	@echo "  make help               Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  # Run with your own files:"
	@echo "  docker run --rm -v \$$(pwd)/my_data:/app/data/input -v \$$(pwd)/results:/app/data/output immuno-pipeline \\"
	@echo "    --fasta /app/data/input/my_protein.fasta \\"
	@echo "    --pdb /app/data/input/my_structure.pdb \\"
	@echo "    --alleles \"HLA-DRB1*03:01,HLA-DRB1*01:01\""

build:
	@echo "Building Docker image..."
	@docker build -t immuno-pipeline .

run: build
	@echo "Running pipeline with example data..."
	@docker run --rm --name immuno-container \
		-v "$(shell pwd)/data/input:/app/data/input" \
		-v "$(shell pwd)/data/output:/app/data/output" \
		immuno-pipeline \
		--fasta /app/data/input/rav_pal.fasta \
		--pdb /app/data/input/3CZO.pdb \
		--alleles "HLA-DRB1*03:01,HLA-DRB1*01:01"

run-custom: build
	@echo "Running pipeline with custom arguments..."
	@echo "Usage: make run-custom FASTA=/path/to/file.fasta PDB=/path/to/file.pdb ALLELES=\"HLA-DRB1*03:01\""
	@docker run --rm --name immuno-container \
		-v "$(shell pwd)/data/input:/app/data/input" \
		-v "$(shell pwd)/data/output:/app/data/output" \
		immuno-pipeline \
		--fasta $(FASTA) \
		$(if $(PDB),--pdb $(PDB),) \
		--alleles "$(ALLELES)"

stop:
	@docker stop immuno-container || true

clean: stop
	@docker rmi immuno-pipeline || true