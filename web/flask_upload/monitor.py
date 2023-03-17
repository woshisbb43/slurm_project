import requests
from prometheus_client.parser import text_string_to_metric_families

def get_metrics():
    # Make a request to the Prometheus metrics endpoint
    metrics = requests.get("http://localhost:8080/metrics").content.decode('utf-8')

    metric_data = {}
    bad_words = ['slurm_nodes_idle', 'slurm_nodes_alloc', 'slurm_partition_cpus_idle', 'slurm_partition_cpus_total', 'slurm_queue_pending', 'slurm_queue_running', 'job', 'slurm_cpus']

    # Iterate over metric families
    for family in text_string_to_metric_families(metrics):
        # Extract metric name and help text
        name = family.name
        if any([x in name for x in bad_words]):
            help_text = family.documentation

            # Initialize dictionary for this metric family
            metric_family_data = {"name": name, "help": help_text, "samples": []}

            # Iterate over metric samples
            for sample in family.samples:
                # Extract sample data
                sample_name = sample.name
                sample_labels = sample.labels
                sample_value = sample.value

                # Add sample data to metric family dictionary
                metric_family_data["samples"].append({"name": sample_name.replace("_", " "), "labels": sample_labels, "value": sample_value})

            # Add metric family dictionary to overall metric data dictionary
            metric_data[name] = metric_family_data

    return metric_data