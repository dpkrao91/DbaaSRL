PYTHON ?= python
SEED   ?= 42
EPISODES ?= 100
EPLEN  ?= 720

.PHONY: all data train figures benchmark paper test clean

all: data train figures benchmark paper

data:
	$(PYTHON) src/workload_generator.py --days 7 --out data --seed $(SEED)
	$(PYTHON) src/workload_generator.py --days 3 --out data_eval --seed 137
	cp data_eval/workload_aggregate.csv data/workload_aggregate_eval.csv
	cp data_eval/workload_combined.csv  data/workload_combined_eval.csv
	rm -rf data_eval

train:
	$(PYTHON) src/train.py \
		--workload data/workload_aggregate.csv \
		--eval-workload data/workload_aggregate_eval.csv \
		--out results --episodes $(EPISODES) --episode-length $(EPLEN) --seed $(SEED)

figures:
	$(PYTHON) src/plot_results.py --results results \
		--workload data/workload_aggregate.csv --out figures

benchmark:
	$(PYTHON) src/benchmark_inference.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	            pdflatex -interaction=nonstopmode main.tex

test:
	$(PYTHON) tests/test_smoke.py

clean:
	rm -rf results/* figures/*.png paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
