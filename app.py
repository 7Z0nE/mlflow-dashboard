import os
import hashlib
from datetime import datetime
import time
from flask import Flask, render_template, request
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def format_relative_time(ts_ms):
    if not ts_ms:
        return "N/A"
    diff_sec = max(0, (time.time() * 1000 - ts_ms) / 1000.0)
    if diff_sec < 60:
        return f"{int(diff_sec)}s ago"
    elif diff_sec < 3600:
        return f"{int(diff_sec // 60)}m ago"
    elif diff_sec < 86400:
        return f"{int(diff_sec // 3600)}h ago"
    else:
        return f"{int(diff_sec // 86400)}d ago"

def format_duration(duration_ms):
    if duration_ms is None or duration_ms < 0:
        return "N/A"
    sec = duration_ms / 1000.0
    if sec < 60:
        return f"{int(sec)}s"
    elif sec < 3600:
        return f"{int(sec // 60)}m {int(sec % 60)}s"
    elif sec < 86400:
        return f"{int(sec // 3600)}h {int((sec % 3600) // 60)}m"
    else:
        return f"{int(sec // 86400)}d {int((sec % 86400) // 3600)}h"

def get_color_and_text_color_from_string(s):
    """Generate a consistent hex color and a contrasting text color from a string."""
    hash_obj = hashlib.md5(s.encode('utf-8'))
    hex_color = hash_obj.hexdigest()[:6]
    
    # Calculate luminance to determine text color
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    text_color = "#000000" if luminance > 0.5 else "#ffffff"
    return f"#{hex_color}", text_color

@app.route('/run/<run_id>/last_active')
def get_run_last_active(run_id):
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        start_ts = run.info.start_time
        last_active_ts = start_ts
        
        if run.info.status == 'RUNNING' and run.data.metrics:
            latest_metric_ts = 0
            # Limit to 10 metrics to prevent slow responses or timeouts
            metric_keys = list(run.data.metrics.keys())[:10]
            for m_key in metric_keys:
                try:
                    history = client.get_metric_history(run_id, m_key)
                    if history and history[-1].timestamp > latest_metric_ts:
                        latest_metric_ts = history[-1].timestamp
                except Exception:
                    pass
            if latest_metric_ts > 0:
                last_active_ts = latest_metric_ts
        
        last_active_str = format_relative_time(last_active_ts)
        
        duration_ms = last_active_ts - start_ts if start_ts else 0
        duration_str = format_duration(duration_ms)
        is_recent = (time.time() * 1000 - last_active_ts) < 5 * 60 * 1000
        
        return {
            "last_active": last_active_str,
            "duration": duration_str,
            "is_recent": is_recent
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/abort/<run_id>', methods=['POST'])
def abort_run(run_id):
    client = MlflowClient()
    try:
        client.set_terminated(run_id, status="KILLED")
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/')
def index():
    client = MlflowClient()
    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    filter_user = request.args.get('user', '')
    filter_exp = request.args.get('experiment', '')
    try:
        experiments = client.search_experiments()
    except Exception as e:
        return f"Error connecting to MLflow: {str(e)}"
        
    experiment_ids = [exp.experiment_id for exp in experiments]
    
    # fetch up to 1000 recent runs across all experiments to allow manual pagination and filtering
    try:
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            order_by=["start_time DESC"],
            max_results=1000
        )
    except Exception as e:
        runs = []
    
    exp_map = {exp.experiment_id: exp.name for exp in experiments}
    
    all_users = set()
    all_experiments = set(exp_map.values())
    
    filtered_runs = []
    for run in runs:
        tags = run.data.tags
        user = tags.get('mlflow.user', 'Unknown')
        user_short = user.split('@')[0] if '@' in user else user
        all_users.add(user_short)
        
        exp_name = exp_map.get(run.info.experiment_id, 'Unknown')
        
        if filter_user and filter_user != user_short:
            continue
        if filter_exp and filter_exp != exp_name:
            continue
            
        filtered_runs.append((run, user_short, exp_name))

    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    total_runs = len(filtered_runs)
    total_pages = max(1, (total_runs + limit - 1) // limit)
    page = max(1, min(page, total_pages))
    
    page_runs = filtered_runs[(page-1)*limit : page*limit]
    
    run_data = []
    for run, user_short, exp_name in page_runs:
        tags = run.data.tags
        start_time_ms = run.info.start_time
        end_time_ms = run.info.end_time
        
        start_time_full = datetime.fromtimestamp(start_time_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S') if start_time_ms else "N/A"
        start_time_rel = format_relative_time(start_time_ms) if start_time_ms else "N/A"
        
        status = run.info.status
        
        last_known_active_ts = end_time_ms if end_time_ms else start_time_ms
        last_known_active_rel = format_relative_time(last_known_active_ts) if last_known_active_ts else "N/A"
        
        is_recent = False
        if last_known_active_ts and (time.time() * 1000 - last_known_active_ts) < 5 * 60 * 1000:
            is_recent = True
        
        if end_time_ms and start_time_ms:
            duration_str = format_duration(end_time_ms - start_time_ms)
        elif status == 'RUNNING':
            duration_str = "⏳"
        else:
            duration_str = "N/A"
            
        source = tags.get('mlflow.source.name', 'Unknown')
        short_source = source.split('/')[-1] if '/' in source else source
        short_source = short_source.split('\\')[-1] if '\\' in short_source else short_source
        
        exp_bg, exp_text = get_color_and_text_color_from_string(exp_name)
        user_bg, user_text = get_color_and_text_color_from_string(user_short)
        
        run_data.append({
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'experiment_name': exp_name,
            'exp_bg': exp_bg,
            'exp_text': exp_text,
            'time_rel': start_time_rel,
            'time_full': start_time_full,
            'duration': duration_str,
            'last_active_rel': last_known_active_rel,
            'is_recent': is_recent,
            'user': user_short,
            'user_bg': user_bg,
            'user_text': user_text,
            'source': short_source,
            'status': status,
        })
        
    return render_template('index.html', 
                           runs=run_data, 
                           all_users=sorted(list(all_users)), 
                           all_experiments=sorted(list(all_experiments)),
                           current_user=filter_user,
                           current_exp=filter_exp,
                           mlflow_url=mlflow_url,
                           page=page,
                           limit=limit,
                           total_pages=total_pages)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
