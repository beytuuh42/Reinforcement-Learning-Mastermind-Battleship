docker build -t rl/tf_agents_with_reverb:2.9.1 .
docker run --name tfagents-reverb -p 8888:8888 -p 6006-6009:6006-6009 rl/tf_agents_with_reverb:2.9.1