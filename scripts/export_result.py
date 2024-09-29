import os
import csv
import argparse
import pandas as pd

def main(result_path, output_path, device_name):
    df_summary = pd.read_csv(os.path.join(result_path, "summary.csv"))
    df_engine = pd.read_csv(os.path.join(result_path, "engine_config.csv"))
    df_system = pd.read_csv(os.path.join(result_path, "system_metrics.csv"))
    df = pd.merge(df_summary, df_engine, on='engine_config_id', how='inner')
    df = pd.merge(df, df_system, on='run_id', how='inner')

    #add device name to the dataframe. Take device name from args
    if device_name:
        df['device_name'] = device_name
    else:
        df['device_name'] = "unknown"
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True, help="The folder containing the results to export.")
    parser.add_argument("--output-path", type=str, required=True, help="The path to export the results to.")
    parser.add_argument("--device-name", type=str, required=False, help="The name of the device to export the results for.")
    args = parser.parse_args()
    main(args.result_path, args.output_path, args.device_name)