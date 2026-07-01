import torch
import subprocess
import json
import os

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

def run_compute_scan(output_file="compute_scan.json"):
    """
    Runs a compute scan using nvidia-smi and torch.cuda.get_device_properties
    to capture current compute capabilities and multi-GPU availability.
    Writes the result to a JSON file.
    """
    scan_result = {
        "device_count": 0,
        "devices": [],
        "nvidia_smi_raw": ""
    }
    
    if torch.cuda.is_available():
        scan_result["device_count"] = torch.cuda.device_count()
        for i in range(scan_result["device_count"]):
            props = torch.cuda.get_device_properties(i)
            scan_result["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count
            })
            
    try:
        smi_output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        scan_result["nvidia_smi_raw"] = smi_output
    except Exception as e:
        scan_result["nvidia_smi_raw"] = f"Failed to run nvidia-smi: {str(e)}"
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scan_result, f, indent=4)
        
    return scan_result
