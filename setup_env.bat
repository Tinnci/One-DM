@echo off
echo 正在创建Python 3.12虚拟环境...

:: 删除旧的虚拟环境（如果存在）
if exist .venv (
    echo 删除旧的虚拟环境...
    rd /s /q .venv
)

:: 使用uv创建Python 3.12的虚拟环境
uv venv --python=3.12 .venv

:: 激活虚拟环境
call .venv\Scripts\activate.bat

:: 使用uv sync安装项目依赖
echo 安装项目依赖...
uv sync

echo 环境设置完成！
echo 使用 .venv\Scripts\activate.bat 激活环境 