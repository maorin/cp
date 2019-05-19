# -*- coding: utf-8 -*-
import sys

if "ssq" == sys.argv[1]:
    from pocket.ssq import ssq
    ssq.main()

if "stock" == sys.argv[1]:
    from pocket.stock import stock
    stock.main()
