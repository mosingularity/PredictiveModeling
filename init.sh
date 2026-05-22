#!/bin/bash
echo "export http_proxy=http://172.28.42.51:8080" >> /etc/environment
echo "export https_proxy=http://172.28.42.51:8080" >> /etc/environment
echo "export NO_PROXY=169.254.169.254,*.azuredatabricks.net,*.blob.core.windows.net,*.dfs.core.windows.net,*.table.core.windows.net,*.queue.core.windows.net,*.service.signalr.net" >> /etc/environment
source /etc/environment