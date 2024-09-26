import akshare as ak
from typing import Literal
import pandas as pd
import numpy as np
from pathlib import Path


class AKData:
    def __init__(self, database_type: Literal["postgres", "h5", "csv"]):
        self.database_type = database_type

    def rewrite(self, func, data_name, *args, **kwargs):
        print(f"rewrite {data_name} data")
        data = func(*args, **kwargs)
        data["symbol"] = kwargs.get("symbol")
        data = self.format_data(data)
        data.to_csv(f"{data_name}.csv", index=False)
        return data

    def local_read_and_append(self, func, data_name: str, rewrite: bool = False):
        def wrapper(*args, **kwargs):
            assert "symbol" in kwargs, "symbol is required"
            assert "start_date" in kwargs, "start_date is required"
            assert "end_date" in kwargs, "end_date is required"
            if (
                self.database_type == "csv"
                and Path(f"{data_name}.csv").exists()
                and not rewrite
            ):
                print(f"load data from {data_name}.csv")
                data = self.load_data_csv(data_name=data_name)
                data = data[data["symbol"] == kwargs.get("symbol")]
                if data.empty:
                    data = self.rewrite(func, data_name, *args, **kwargs)
                data = data[
                    ((data["date"] >= kwargs.get("start_date")))
                    & (data["date"] <= kwargs.get("end_date"))
                ]
                if data.empty:
                    data = self.rewrite(func, data_name, *args, **kwargs)
                # TODO 开发前期只写了一个简单的逻辑，后续需要完善
                exit_end_date = data["date"].max()
                if exit_end_date < kwargs.get("end_date"):
                    kwargs["start_date"] = exit_end_date
                    add_data["symbol"] = kwargs.get("symbol")
                    add_data = self.format_data(func(*args, **kwargs))
                    data = pd.concat([data, add_data])
                    data.to_csv(f"{data_name}.csv", index=False)
            else:
                data = self.rewrite(func, data_name, *args, **kwargs)
            return data

        return wrapper

    def stock_zh_a_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
        rewrite: bool = False,
    ):
        return self.local_read_and_append(
            ak.stock_zh_a_daily,
            data_name="stock_zh_a_daily",
            rewrite=rewrite,
        )(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)

    def format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.empty, "data is empty"
        result = data.copy()
        result.rename(
            columns={
                "日期": "date",
                "开盘价": "open",
                "收盘价": "close",
                "最高价": "high",
                "最低价": "low",
                "成交量": "volume",
                "成交额": "amount",
            },
            inplace=True,
        )
        if "date" in data.columns:
            result["date"] = pd.to_datetime(result["date"])
        for symbol_col in [i for i in data.columns if "symbol" in i]:
            if data[symbol_col].dtype == "int64":
                result[symbol_col] = data[symbol_col].apply(lambda x: str(x).zfill(6))
        return result

    def load_data_h5(self, data_name):
        pass

    def load_data_postgres(self, data_name):
        pass

    def load_data_csv(self, data_name: str):
        data = pd.read_csv(f"{data_name}.csv")
        assert not data.empty, "data is empty"
        data["date"] = pd.to_datetime(data["date"])
        if "symbol" in data.columns:
            if data["symbol"].dtype == "int64":
                data["symbol"] = data["symbol"].apply(lambda x: str(x).zfill(6))
        return data

    # NOTE 股票数据
    def stock_individual_spot_xq(self, symbol: str):
        """
        个股实时行情-雪球数据
        param symbol: 股票代码
            中证: {"CSI000985":中证全指, "CSI932000":"中证2000", "CSI000852":"中证1000","CSI000905":"中证500"}
            上证: {"SH000001":"上证指数", "SH000300":"沪深300"}
            深证: {"SZ399303":"国证2000", "SZ399001":"深证成指", "SZ399006":"创业板指"}
        """
        return ak.stock_individual_spot_xq(symbol=symbol)

    def stock_zh_a_daily_hfq(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        return self.local_read_and_append(
            ak.stock_zh_a_daily, data_name="stock_zh_a_daily_hfq", rewrite=rewrite
        )(symbol=symbol, start_date=start_date, end_date=end_date, adjust="hfq")

    def stock_zh_a_daily_qfq(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        return self.local_read_and_append(
            ak.stock_zh_a_daily, data_name="stock_zh_a_daily_qfq", rewrite=rewrite
        )(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

    def stock_zh_a_daily(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        return self.local_read_and_append(
            ak.stock_zh_a_daily, data_name="stock_zh_a_daily", rewrite=rewrite
        )(symbol=symbol, start_date=start_date, end_date=end_date, adjust="")

    # NOTE 指数数据
    def stock_zh_index_spot_sina(rewrite: bool = False):
        """
        新浪财经-行情中心首页-A股-分类-所有指数
        """
        if Path("stock_zh_index_spot_sina.csv").exists() and not rewrite:
            print("load data from stock_zh_index_spot_sina.csv")
            data = pd.read_csv("stock_zh_index_spot_sina.csv")
        else:
            data = ak.stock_zh_index_spot_sina()
            data.to_csv("stock_zh_index_spot_sina.csv", index=False)
        return data

    def _stock_zh_index_daily_em(
        self, symbol: str, start_date: np.datetime64, end_date: np.datetime64
    ):
        start_date = np.datetime_as_string(start_date, unit="D").replace("-", "")
        end_date = np.datetime_as_string(end_date, unit="D").replace("-", "")
        return ak.stock_zh_index_daily_em(
            symbol=symbol, start_date=start_date, end_date=end_date
        )

    def stock_zh_index_daily(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        """
        历史行情数据-东方财富
        """
        return self.local_read_and_append(
            self._stock_zh_index_daily_em,
            data_name="stock_zh_index_daily",
            rewrite=rewrite,
        )(symbol=symbol, start_date=start_date, end_date=end_date)

    def index_zh_a_hist(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        period: Literal["daily", "weekly", "monthly"] = "daily",
        rewrite: bool = False,
    ):
        """
        东方财富网-中国股票指数-行情数据
        单次返回具体指数指定 period 从 start_date 到 end_date 的之间的近期数据
        """
        return self.local_read_and_append(
            ak.index_zh_a_hist, data_name="index_zh_a_hist", rewrite=rewrite
        )(symbol=symbol, start_date=start_date, end_date=end_date)

    def _stock_zh_index_hist_csindex(
        self, symbol: str, start_date: np.datetime64, end_date: np.datetime64
    ):
        start_date = np.datetime_as_string(start_date, unit="D").replace("-", "")
        end_date = np.datetime_as_string(end_date, unit="D").replace("-", "")
        data = ak.stock_zh_index_hist_csindex(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
        data = data.rename(
            columns={
                "最高": "high",
                "最低": "low",
                "开盘": "open",
                "收盘": "close",
            }
        )
        return data

    def stock_zh_index_hist_csindex(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        """中证指数历史数据
        param
            symbol: 中证指数代码, e.g. "000985","932000","000852","000905"
        """
        return self.local_read_and_append(
            self._stock_zh_index_hist_csindex,
            data_name="stock_zh_index_hist_csindex",
            rewrite=rewrite,
        )(symbol=symbol, start_date=start_date, end_date=end_date)

    def index_stock_info(self, rewrite: bool = False):
        """
        聚宽-指数数据-指数列表
        """
        if Path("index_stock_info.csv").exists() and not rewrite:
            print("load data from index_stock_info.csv")
            data = pd.read_csv("index_stock_info.csv")
            data = data.copy().rename(columns={"index_code	": "symbol"})
            data = self.format_data(data)
        else:
            data = ak.index_stock_info()
            data = data.copy().rename(columns={"index_code": "symbol"})
            self.format_data(data).to_csv("index_stock_info.csv", index=False)
        return data

    # NOTE 通用 指数成分股
    def _index_stock_cons(self, symbol: str):
        data = ak.index_stock_cons(symbol=symbol)
        data = data.copy().rename(columns={"品种代码": "cons_symbol"})
        return data

    def index_stock_cons(self, symbol: str, rewrite: bool = False):
        if Path("index_stock_cons.csv").exists() and not rewrite:
            data = self.format_data(pd.read_csv("index_stock_cons.csv"))
            data = data[data["symbol"] == symbol]
            if data.empty:
                data = self.rewrite(
                    self._index_stock_cons, "index_stock_cons", symbol=symbol
                )
            else:
                print("load data from index_stock_cons.csv")
        else:
            data = self.rewrite(
                self._index_stock_cons, "index_stock_cons", symbol=symbol
            )
        return data

    # NOTE 中证指数成分股
    def _index_stock_cons_csindex(self, symbol: str):
        data = ak.index_stock_cons_csindex(symbol=symbol)
        data = data.copy().rename(
            columns={"指数代码": "symbol", "成分券代码": "cons_symbol"}
        )
        return data

    def index_stock_cons_csindex(self, symbol: str, rewrite: bool = False):
        if Path("index_stock_cons_csindex.csv").exists() and not rewrite:
            data = self.format_data(pd.read_csv("index_stock_cons_csindex.csv"))
            data = data[data["symbol"] == symbol]
            if data.empty:
                data = self.rewrite(
                    self._index_stock_cons_csindex,
                    "index_stock_cons_csindex",
                    symbol=symbol,
                )
            else:
                print("load data from index_stock_cons_csindex.csv")
        else:
            data = self.rewrite(
                self._index_stock_cons_csindex,
                "index_stock_cons_csindex",
                symbol=symbol,
            )
        return data

    def _index_hist_cni(
        self, symbol: str, start_date: np.datetime64, end_date: np.datetime64
    ):
        start_date = np.datetime_as_string(start_date, unit="D").replace("-", "")
        end_date = np.datetime_as_string(end_date, unit="D").replace("-", "")
        data = ak.index_hist_cni(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
        return data

    def index_hist_cni(
        self,
        symbol: str,
        start_date: np.datetime64,
        end_date: np.datetime64,
        rewrite: bool = False,
    ):
        return self.local_read_and_append(
            self._index_hist_cni, data_name="index_hist_cni", rewrite=rewrite
        )(symbol=symbol, start_date=start_date, end_date=end_date)

    def futures_display_main_sina(self, rewrite: bool = False):
        if Path("futures_display_main_sina.csv").exists() and not rewrite:
            print("load data from futures_display_main_sina.csv")
            data = pd.read_csv("futures_display_main_sina.csv")
        else:
            data = ak.futures_display_main_sina()
            data.to_csv("futures_display_main_sina.csv", index=False)
        return data

    def futures_main_sina(
        self, symbol: str, start_date: str, end_date: str, rewrite: bool = False
    ):
        return self.local_read_and_append(
            ak.futures_main_sina,
            data_name="futures_main_sina",
            rewrite=rewrite,
        )(symbol=symbol, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    ak_data = AKData(database_type="csv")
    df = ak_data.index_stock_cons(symbol="399303")
    # print(df)
