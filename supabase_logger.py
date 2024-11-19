from supabase import create_client
import numpy as np
import torch

# Supabase configuration
SUPABASE_URL = "https://xqvexgqyezobttcqdspe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhxdmV4Z3F5ZXpvYnR0Y3Fkc3BlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5ODQ1MDgsImV4cCI6MjA0NzU2MDUwOH0.aevAXcf6Rl5xDshitXETdUu2j895bZlrDb6lD2ZwhlI"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def tensor_to_list(tensor):
    """Convert a PyTorch tensor to a Python list."""
    return tensor.detach().cpu().numpy().tolist()


def log_experiment_result(
        dataset,
        autoencoder_notes,
        original_x,
        original_y,
        digit_number,
        auto_prediction_label,
        mlp_prediction_label,
        num_confused,
        includes_true,
        target_distribution,
        autoadvex_x_hat,
        autoadvex_y_hat,
        autoadvex_label_kld,
        autoadvex_mse,
        autoadvex_frob,
        mlpadvex_x_hat,
        mlpadvex_y_hat,
        mlpadvex_label_kld,
        mlpadvex_mse,
        mlpadvex_frob
):
    """Log a single experiment result to Supabase."""
    data = {
        "dataset": dataset,
        "autoencoder_notes": autoencoder_notes,
        "original_x": tensor_to_list(original_x),
        "original_y": tensor_to_list(original_y),
        "digit_number": float(digit_number),
        "auto_prediction_label": tensor_to_list(auto_prediction_label),
        "mlp_prediction_label": tensor_to_list(mlp_prediction_label),
        "target_distribution": tensor_to_list(target_distribution),
        "num_confused": float(num_confused),
        "includes_true": bool(includes_true),
        "autoadvex_x_hat": tensor_to_list(autoadvex_x_hat),
        "autoadvex_y_hat": tensor_to_list(autoadvex_y_hat),
        "autoadvex_label_kld": float(autoadvex_label_kld),
        "autoadvex_mse": float(autoadvex_mse),
        "autoadvex_frob": float(autoadvex_frob),
        "mlpadvex_x_hat": tensor_to_list(mlpadvex_x_hat),
        "mlpadvex_y_hat": tensor_to_list(mlpadvex_y_hat),
        "mlpadvex_label_kld": float(mlpadvex_label_kld),
        "mlpadvex_mse": float(mlpadvex_mse),
        "mlpadvex_frob": float(mlpadvex_frob)
    }

    try:
        result = supabase.table("autoadvex_results").insert(data).execute()
        print("Successfully logged experiment result")
        return True
    except Exception as e:
        print(f"Error logging result: {str(e)}")
        return False

