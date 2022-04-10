---
title: Local
icon: material/folder-open
---

# Local

Repeatedly hitting remote API endpoints is costly, thus it's very important to cache data locally. 
Luckily, vectorbt implements a range of ways for managing local data.

## Pickling

Like any other class subclassing [Pickleable](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable), 
we can save any [Data](/api/data/base/#vectorbtpro.data.base.Data) instance to the disk using 
[Pickleable.save](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable.save) and load it back 
using [Pickleable.load](/api/utils/pickling/#vectorbtpro.utils.pickling.Pickleable.load). This will pickle
the entire Python object including the stored Pandas objects, symbol dictionaries, and settings:

```pycon
>>> import vectorbtpro as vbt

>>> yf_data = vbt.YFData.fetch(
...     ['BTC-USD', 'ETH-USD'], 
...     start='2020-01-01', 
...     end='2020-01-05')
```

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> yf_data.save('yf_data')  # (1)!

>>> yf_data = vbt.YFData.load('yf_data')  # (2)!
>>> yf_data = yf_data.update(end='2020-01-06')
>>> yf_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
2020-01-05 00:00:00+00:00  7411.317383  136.276779
```

1. Automatically adds the extension `.pickle` to the file name
2. The object can be loaded back in a new runtime or even on another machine, just make
sure to use a compatible vectorbt version

!!! important
    The class definition won't be saved. If a new version of vectorbt introduces a breaking change
    to the [Data](/api/data/base/#vectorbtpro.data.base.Data) constructor, the object may not load. 
    In such a case, you can manually create a new instance: 

    ```pycon
    >>> yf_data = vbt.YFData(**vbt.Configured.load('yf_data').config)
    ```

## Saving

While pickling is a pretty fast and convenient solution to storing Python objects of any size, the 
pickled file is effectively a black box that requires a Python interpreter to be unboxed, which makes 
it unusable for many tasks since it cannot be imported by most other data-driven tools. To lift this
limitation, the [Data](/api/data/base/#vectorbtpro.data.base.Data) class allows us for saving 
exclusively the stored Pandas objects into one to multiple files of a tabular format.

### CSV

The first supported file format is the [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) format,
which is implemented by the instance method [Data.to_csv](/api/data/base/#vectorbtpro.data.base.Data.to_csv).
This method takes a path to the directory where the data should be stored (`dir_path`), and saves each symbol 
in a separate file using [DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).

!!! info
    Multiple symbols cannot be stored inside a single CSV file.

By default, it appends the extension `.csv` to each symbol, and saves the files into the current directory:

```pycon
>>> yf_data.to_csv()
```

We can list all CSV files in the current directory using [pathlib](https://realpython.com/python-pathlib/):

```pycon
>>> from pathlib import Path

>>> list(Path.cwd().glob('*.csv'))
[PosixPath('/Users/olegpolakow/Documents/GitHub/vectorbt.pro/ETH-USD.csv'),
 PosixPath('/Users/olegpolakow/Documents/GitHub/vectorbt.pro/BTC-USD.csv')]
```

A cleaner approach is to save all the data in a separate directory:

```pycon
>>> (Path.cwd() / 'ETH-USD.csv').unlink()  # (1)!
>>> (Path.cwd() / 'BTC-USD.csv').unlink()

>>> yf_data.to_csv('data', mkdir_kwargs=dict(mkdir=True))  # (2)!
```

1. Delete the CSV files created previously from the current directory
2. Save the files to the directory with the name `data`. If the directory doesn't exist, create a new one
by passing keyword arguments `mkdir_kwargs` down to 
[check_mkdir](/api/utils/path_/#vectorbtpro.utils.path_.check_mkdir).

To save the data as tab-separated values (TSV):

```pycon
>>> yf_data.to_csv('data', ext='tsv')

>>> list((Path.cwd() / 'data').glob('*.tsv'))
[PosixPath('/Users/olegpolakow/Documents/GitHub/vectorbt.pro/data/BTC-USD.tsv'),
 PosixPath('/Users/olegpolakow/Documents/GitHub/vectorbt.pro/data/ETH-USD.tsv')]
```

!!! hint
    You don't have to pass `sep`: vectorbt will recognize the extension and pass the correct delimiter.
    But you can still override this argument if you need to split the data by a special character.

Similarly to [Data.fetch](/api/data/base/#vectorbtpro.data.base.Data.fetch), we can provide
any argument as a symbol dictionary of type [symbol_dict](/api/data/base/#vectorbtpro.data.base.symbol_dict)
to define different rules for different symbols. Let's store the symbols from our example in separate
directories:

```pycon
>>> yf_data.to_csv(
...     vbt.symbol_dict({
...         'BTC-USD': 'btc_data',
...         'ETH-USD': 'eth_data'
...     }), 
...     mkdir_kwargs=dict(mkdir=True)
... )
```

To have a complete control over the name of each file, use the `path_or_buf` argument:

```pycon
>>> yf_data.to_csv(
...     path_or_buf=vbt.symbol_dict({
...         'BTC-USD': 'data/btc_usd.csv',
...         'ETH-USD': 'data/eth_usd.csv'
...     }), 
...     mkdir_kwargs=dict(mkdir=True)
... )
```

To delete the entire directory (as part of a clean-up, for example):

```pycon
>>> import shutil

>>> shutil.rmtree('data')
```

### HDF

The second supported file format is the [HDF](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format,
which is implemented by the instance method [Data.to_hdf](/api/data/base/#vectorbtpro.data.base.Data.to_hdf).
In contrast to [Data.to_csv](/api/data/base/#vectorbtpro.data.base.Data.to_csv), this method can store
multiple symbols inside a single file, where symbols are distributed as HDF keys. The first argument is not a 
path to a directory anymore, but a path to a file (`file_path`).

By default, it creates a new file with the same name as the name of the data class and an extension
`.h5`, and saves each symbol under a separate key in that file using 
[DataFrame.to_hdf](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_hdf.html):

```pycon
>>> yf_data.to_hdf()

>>> list(Path.cwd().glob('*.h5'))
[PosixPath('/Users/olegpolakow/Documents/GitHub/vectorbt.pro/YFData.h5')]
```

To see the list of all the groups and keys contained in an HDF file:

```pycon
>>> import pandas as pd

>>> with pd.HDFStore('YFData.h5') as store:
...     print(store.keys())
['/BTC-USD', '/ETH-USD']
```

Use the `key` argument to manually specify the key of a particular symbol:

```pycon
>>> yf_data.to_hdf(
...     key=vbt.symbol_dict({
...         'BTC-USD': 'btc_usd',
...         'ETH-USD': 'eth_usd'
...     })
... )
```

!!! hint
    If there is only one symbol, you don't need to use [symbol_dict](/api/data/base/#vectorbtpro.data.base.symbol_dict),
    just pass `key='btc_usd'`.

Use the `path_or_buf` argument to store symbols across multiple HDF files:

```pycon
>>> yf_data.to_hdf(
...     path_or_buf=vbt.symbol_dict({
...         'BTC-USD': 'btc_usd.h5',
...         'ETH-USD': 'eth_usd.h5'
...     })
... )
```

The arguments `path_or_buf` and `key` can be combined.

Any other argument behaves the same as for 
[Data.to_csv](/api/data/base/#vectorbtpro.data.base.Data.to_csv).

## Loading

To import any previously stored data in a tabular format, we can either use Pandas or vectorbt's preset 
data classes specifically crafted for this job.

### CSV

Each CSV dataset can be manually imported using 
[pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html):

```pycon
>>> yf_data.to_csv()

>>> pd.read_csv('BTC-USD.csv', index_col=0, parse_dates=True)
                                  Open         High          Low        Close  \\
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112          0             0  
2020-01-01 00:00:00+00:00  18565664997          0             0  
2020-01-02 00:00:00+00:00  20802083465          0             0  
2020-01-03 00:00:00+00:00  28111481032          0             0  
2020-01-04 00:00:00+00:00  18444271275          0             0  
2020-01-05 00:00:00+00:00  19725074095          0             0 
```

To join the imported datasets and wrap them with [Data](/api/data/base/#vectorbtpro.data.base.Data), 
we can use [Data.from_data](/api/data/base/#vectorbtpro.data.base.Data.from_data):

```pycon
>>> btc_usd = pd.read_csv('BTC-USD.csv', index_col=0, parse_dates=True)
>>> eth_usd = pd.read_csv('ETH-USD.csv', index_col=0, parse_dates=True)

>>> data = vbt.Data.from_data({'BTC-USD': btc_usd, 'ETH-USD': eth_usd})
>>> data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
2020-01-05 00:00:00+00:00  7411.317383  136.276779
```

To relieve the user of the burden of manually searching, fetching, and merging CSV data, vectorbt implements 
the class [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData), which can recursively explore directories
for CSV files, resolve path expressions using [glob](https://docs.python.org/3/library/glob.html), translate 
the found paths into symbols, and import and join tabular data - all automatically via a single command. 
It subclasses another class - [LocalData](/api/data/custom/#vectorbtpro.data.custom.LocalData), which takes
the whole credit for making all the above possible. 

At the heart of the path matching functionality is the class method 
[LocalData.fetch](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch), which
iterates over the specified paths, and for each one, finds the matching absolute paths using 
another class method [LocalData.match_path](/api/data/custom/#vectorbtpro.data.custom.LocalData.match_path), 
and calls the abstract class method [LocalData.fetch_symbol](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch_symbol)
to pull the data from the file located under that path.

Let's explore how [LocalData.match_path](/api/data/custom/#vectorbtpro.data.custom.LocalData.match_path) works
by creating a directory with the name `data` and storing various empty files inside:

```pycon
>>> from vectorbtpro.utils.path_ import tree

>>> Path('data').mkdir(exist_ok=True)
>>> (Path('data') / 'file1.csv').touch()
>>> (Path('data') / 'file2.tsv').touch()
>>> (Path('data') / 'file3').touch()
>>> (Path('data') / 'sub-data').mkdir(exist_ok=True)
>>> (Path('data') / 'sub-data' / 'file1.csv').touch()
>>> (Path('data') / 'sub-data' / 'file2.tsv').touch()
>>> (Path('data') / 'sub-data' / 'file3').touch()

>>> print(tree('data'))
data
├── file1.csv
├── file2.tsv
├── file3
└── sub-data
    ├── file1.csv
    ├── file2.tsv
    └── file3

1 directories, 6 files
```

Match all files in a directory:

```pycon
>>> vbt.LocalData.match_path('data')
[PosixPath('data/file1.csv'),
 PosixPath('data/file2.tsv'),
 PosixPath('data/file3')]
```

Match all CSV files in a directory:

```pycon
>>> vbt.LocalData.match_path('data/*.csv')
[PosixPath('data/file1.csv')]
```

Match all CSV files in a directory recursively:

```pycon
>>> vbt.LocalData.match_path('data/**/*.csv')
[PosixPath('data/file1.csv'), PosixPath('data/sub-data/file1.csv')]
```

For more details, see the documentation of [glob](https://docs.python.org/3/library/glob.html).

Going back to [LocalData.fetch](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch): it can match
one or multiple path expressions like above, provided either as `symbols` (if `paths` is None) or `paths`.
Whenever we provide paths as symbols, the method calls 
[LocalData.path_to_symbol](/api/data/custom/#vectorbtpro.data.custom.LocalData.path_to_symbol) on each
matched path to parse the name of the symbol (by default, it's the stem of the path):

```pycon
>>> vbt.CSVData.fetch('BTC-USD.csv').symbols
['BTC-USD']

>>> vbt.CSVData.fetch(['BTC-USD.csv', 'ETH-USD.csv']).symbols
['BTC-USD', 'ETH-USD']

>>> vbt.CSVData.fetch('*.csv').symbols
['BTC-USD', 'ETH-USD']

>>> vbt.CSVData.fetch(['BTC/USD', 'ETH/USD'], paths='*.csv').symbols  # (1)!
['BTC/USD', 'ETH/USD']

>>> vbt.CSVData.fetch(  # (2)!
...     ['BTC/USD', 'ETH/USD'], 
...     paths=['BTC-USD.csv', 'ETH-USD.csv']
... ).symbols
['BTC/USD', 'ETH/USD']
```

1. Specify the symbols explicitly
2. Specify the symbols and the paths explicitly

!!! note
    Don't forget to filter by the `.csv`, `.tsv`, or any other extension in the expression.

Whenever we use a wildcard such as `*.csv`, vectorbt will sort the matched paths (per each path 
expression). To disable sorting, set `sort_paths` to False. We can also disable the path matching 
mechanism entirely by setting `match_paths` to False, which will forward all arguments directly to 
[CSVData.fetch_symbol](/api/data/custom/#vectorbtpro.data.custom.CSVData.fetch_symbol):

```pycon
>>> vbt.CSVData.fetch(
...     ['BTC/USD', 'ETH/USD'], 
...     paths=vbt.symbol_dict({
...         'BTC/USD': 'BTC-USD.csv',
...         'ETH/USD': 'ETH-USD.csv'
...     }),
...     match_paths=False
... ).symbols
['BTC/USD', 'ETH/USD']
```

!!! hint
    Instead of paths, you can pass objects of any type supported by the `filepath_or_buffer` argument in 
    [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

To sum up the techniques discussed above, let's create an empty directory with the name `data` (again),
write the `BTC-USD` symbol to a CSV file and the `ETH-USD` symbol to a TSV file, and load both
datasets with a single `fetch` call:

```pycon
>>> if Path('data').exists():
...     shutil.rmtree('data')

>>> yf_data.to_csv(
...     'data',
...     ext=vbt.symbol_dict({
...         'BTC-USD': 'csv',
...         'ETH-USD': 'tsv'
...     }), 
...     mkdir_kwargs=dict(mkdir=True)
... )

>>> csv_data = vbt.CSVData.fetch(['data/*.csv', 'data/*.tsv'])  # (1)!
```

1. The delimiter is recognized automatically based on the file's extension

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> csv_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
2020-01-05 00:00:00+00:00  7411.317383  136.276779
```

!!! note
    Providing two paths with wildcards (`*`) doesn't mean we will get exactly two symbols: 
    there may be more than one path matching each wildcard. You should imagine the two expressions
    above as being combined using the OR rule into a single expression `data/*.{csv,tsv}` 
    (which isn't supported by [glob](https://docs.python.org/3/library/glob.html), unfortunately).

The last but not the least is regex matching with `match_regex`, which instructs vectorbt to 
iterate over all matched paths and additionally validate them against a regular expression:

```pycon
>>> vbt.CSVData.fetch(
...     'data/**/*',  # (1)!
...     match_regex=r'^.*\.(csv|tsv)$'  # (2)!
... ).get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
2020-01-05 00:00:00+00:00  7411.317383  136.276779
```

1. Recursively get the paths of all files in all subdirectories in `data`
2. Filter out any paths that do not end with `csv` or `tsv`

Any other argument is being passed directly to 
[CSVData.fetch_symbol](/api/data/custom/#vectorbtpro.data.custom.CSVData.fetch_symbol) and then
to [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).

#### Chunking

As an alternative to reading everything into memory, Pandas allows us to read data in chunks. 
In the case of CSV, we can load only a subset of lines into memory at any given time.
Even though this is a very useful concept for processing big data, chunking doesn't provide many
benefits when the only goal is to load the entire data into memory anyway.

Where chunking becomes really useful though is data filtering! The class 
[CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) as well as the function
[pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) it's based on 
don't have arguments for skipping rows based on their content, only based on their row index. For example, 
to skip all the data that comes before `2020-01-03`, we would need to load the entire data into memory first.
But once data becomes too large, we may run out of RAM. To account for this, we can split
data into chunks and check the condition on each chunk at a time.

We have two options from here:

1. Use `chunksize` to split data into chunks of a fixed length
2. Use `iterator` to return an iterator that can be used to read chunks of a variable length

Both options make [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
return an iterator of type `TextFileReader`. To make use of this iterator, 
[CSVData.fetch_symbol](/api/data/custom/#vectorbtpro.data.custom.CSVData.fetch_symbol)
accepts a user-defined function `chunk_func` that should 1) accept the iterator, 2) select, process, and 
concatenate chunks, and 3) return a Series or a DataFrame. 

Let's fetch only those rows that have the date ending with an even day:

```pycon
>>> csv_data = vbt.CSVData.fetch(
...     ['data/*.csv', 'data/*.tsv'],
...     chunksize=1,  # (1)!
...     chunk_func=lambda iterator: pd.concat([
...         df 
...         for df in iterator
...         if (df.index.day % 2 == 0).all()
...     ], axis=0)
... )
```

1. Each chunk will be a DataFrame with one row

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> csv_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

!!! note
    Chunking should mainly be used when memory considerations are more important than speed considerations.

### HDF

Each HDF dataset can be manually imported using 
[pandas.read_hdf](https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html):

```pycon
>>> yf_data.to_hdf()

>>> pd.read_hdf('YFData.h5', key='BTC-USD')
                                  Open         High          Low        Close  \\
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112          0             0  
2020-01-01 00:00:00+00:00  18565664997          0             0  
2020-01-02 00:00:00+00:00  20802083465          0             0  
2020-01-03 00:00:00+00:00  28111481032          0             0  
2020-01-04 00:00:00+00:00  18444271275          0             0  
2020-01-05 00:00:00+00:00  19725074095          0             0 
```

Similarly to [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) for CSV data, vectorbt
implements a preset class [HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) tailored
for reading HDF files. It shares the same parent class [LocalData](/api/data/custom/#vectorbtpro.data.custom.LocalData)
and its fetcher [LocalData.fetch](/api/data/custom/#vectorbtpro.data.custom.LocalData.fetch).
But in contrast to CSV datasets, which are stored one per file, HDF datasets are stored one per key
in an HDF file. Since groups and keys follow the [POSIX](https://en.wikipedia.org/wiki/POSIX)-style 
hierarchy with `/`-separators, we can query them in the same way as we query directories and files 
in a regular file system. Read more in [Groups and hierarchical organization](Groups and hierarchical organization).

Let's illustrate this by using [HDFData.match_path](/api/data/custom/#vectorbtpro.data.custom.HDFData.match_path),
which upgrades [LocalData.match_path](/api/data/custom/#vectorbtpro.data.custom.LocalData.match_path)
with a proper discovery and handling of HDF groups and keys:

```pycon
>>> vbt.HDFData.match_path('YFData.h5')
[PosixPath('YFData.h5/BTC-USD'), PosixPath('YFData.h5/ETH-USD')]
```

As we can see, the HDF file above is now being treated as a directory while groups and keys are being treated 
as subdirectories and files respectively. This makes importing HDF datasets as easy as CSV datasets:

```pycon
>>> vbt.HDFData.fetch('YFData.h5/BTC-USD').symbols  # (1)!
['BTC-USD']

>>> vbt.HDFData.fetch('YFData.h5').symbols  # (2)!
['BTC-USD', 'ETH-USD']

>>> vbt.HDFData.fetch('YFData.h5/BTC*').symbols  # (3)!
['BTC-USD']

>>> vbt.HDFData.fetch('*.h5/BTC-*').symbols  # (4)!
['BTC-USD']
```

1. Matches the key `BTC-USD` in `YFData.h5`
2. Matches all keys in `YFData.h5`
3. Matches all keys starting with `BTC` in `YFData.h5` 
4. Matches all keys starting with `BTC` in all HDF files

Any other argument behaves the same as for [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData),
but now it's being passed directly to [HDFData.fetch_symbol](/api/data/custom/#vectorbtpro.data.custom.HDFData.fetch_symbol) 
and then to [pandas.read_hdf](https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html).

#### Chunking

Chunking for HDF files is identical to that for CSV files, but with two exceptions: the data must be
saved as a [PyTables](https://www.pytables.org/) Table structure by using `format='table'`, and
the iterator is now of type `TableIterator` instead of `TextFileReader`.

```pycon
>>> yf_data.to_hdf(format='table')

>>> hdf_data = vbt.HDFData.fetch(
...     'YFData.h5',
...     chunksize=1,
...     chunk_func=lambda iterator: pd.concat([
...         df 
...         for df in iterator
...         if (df.index.day % 2 == 0).all()
...     ], axis=0)
... )
```

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> hdf_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

## Updating

Tabular data such as CSV and HDF data can be read line by line, which makes possible listening for data updates.
The classes [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) and 
[HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) can be updated like every preset data class 
by keeping track of the last row index in [Data.returned_kwargs](/api/data/base/#vectorbtpro.data.base.Data.returned_kwargs).
Whenever an update is triggered, this index is being used as the start row from which the dataset
should be read. After the update, the end row is being used as the new last row index.

Let's separately download the data for `BTC-USD` and `ETH-USD`, save them to one HDF file, and
read the entire file using [HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData):

```pycon
>>> yf_data_btc = vbt.YFData.fetch(
...     'BTC-USD', 
...     start='2020-01-01', 
...     end='2020-01-03')
>>> yf_data_eth = vbt.YFData.fetch(
...     'ETH-USD', 
...     start='2020-01-03', 
...     end='2020-01-05')

>>> yf_data_btc.to_hdf('data.h5', key='yf_data_btc')
>>> yf_data_eth.to_hdf('data.h5', key='yf_data_eth')

>>> hdf_data = vbt.HDFData.fetch(['BTC-USD', 'ETH-USD'], paths='data.h5')
```

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> hdf_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00          NaN  134.171707
2020-01-04 00:00:00+00:00          NaN  135.069366
```

Let's look at the last row index in each dataset:

```pycon
>>> hdf_data.returned_kwargs
{'BTC-USD': {'last_row': 2}, 'ETH-USD': {'last_row': 2}}
```

We see that the third row in each dataset is the new start row (1 row holding the header and 
2 rows holding the data). Let's append new data to the `BTC-USD` dataset and then update our 
[HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) instance:

```pycon
>>> yf_data_btc = yf_data_btc.update(end='2020-01-06')
>>> yf_data_btc.to_hdf('data.h5', key='yf_data_btc')

>>> hdf_data = hdf_data.update()
>>> hdf_data.get('Close')
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
2020-01-05 00:00:00+00:00  7411.317383         NaN
```

The `BTC-USD` dataset has been updated with 3 new data points while the `ETH-USD` dataset hasn't been updated.
This is reflected in the last row index:

```pycon
>>> hdf_data.returned_kwargs
{'BTC-USD': {'last_row': 5}, 'ETH-USD': {'last_row': 2}}
```

This workflow can be repeated endlessly.