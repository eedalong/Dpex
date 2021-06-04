from setuptools import setup
import setuptools


def read(fname):
    with open(fname, 'r') as f:
        return f.read()


setup(
      name             = 'Dpex',
      description      = 'Pytorch Distributed DataLoader Based On Ray',
      long_description = read('README.md'),
      long_description_content_type="text/markdown",
      packages         = setuptools.find_packages(),
      version          = '1.0',
      author           = 'Xiulong Yuan, Zhan Lu, Zheng Zeng, Wenxuan Ma',
      author_email     = 'yuanxl19@mails.tsinghua.edu.cn, lu-z18@mails.tsinghua.edu.cn, zengz17@mails.tsinghua.edu.cn, mwx18@mails.tsinghua.edu.cn',
      url              = 'https://github.com/eedalong/Dpex',
      license          = 'License :: OSI Approved :: MIT License',
      platforms        = 'Linux',
      classifiers      = [
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'Intended Audience :: System Administrators',
          'License :: OSI Approved :: MIT License',
          'Operating System :: POSIX',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python'
          ],
      )
