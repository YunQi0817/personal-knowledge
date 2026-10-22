"""界面层：Streamlit 应用与主题样式。

.. warning::
   本目录里**不能**出现名为 ``app.py`` 的模块。``streamlit run`` 会把脚本所在
   目录加入 ``sys.path``，若存在 ``app.py``，``import app`` 就会解析到那个文件
   而不是 ``app/`` 包，报错 "app is not a package"。
"""

from __future__ import annotations
