[Unit]
Description=Temporal PDF Pipeline Worker
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/husammm/Desktop/courses/cs_courses/RL/projects/Hierarchical_RL_Agents_for_Legal_Contract_Analysis/app/pdf-extraction
ExecStart=/home/husammm/anaconda3/envs/contract_analysis_rl/bin/python worker.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target