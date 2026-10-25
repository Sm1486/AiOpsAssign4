
run python -r requirements.txt
python -m spacy download en_core_web_sm

move into your local directory and run ./node_exporter
move into your local prometheus directory and run ./prometheus --config.file={location of my prometheus.yml file}

streamlit run image_captioner.py

AI Disclaimer
* I used AI to give me the names of the terms to be used in Counter, Gauge, Histogram and Summary functions.
* I used it to generate the PromQLs used for the Histogram plots in granafa.
* I alse used AI to generate the following prompts
- sum by (client_ip) (increase(captioner_images_processed_total[$__range]))
- 100 - (avg(rate(node_cpu_seconds_total{job="node", mode="idle"}[1m]))*100)
- 100*(1 - node_memory_MemAvailable_bytes{job="node"} / node_memory_MemTotal_bytes{job="node"})
- 100 - (node_filesystem_avail_bytes{job="node", mountpoint="/", fstype!~"tmpfs|overlay"} / node_filesystem_size_bytes{job="node", mountpoint="/", fstype!~"tmpfs|overlay"}) * 100
* I also used it for the queries in the Correlation row.