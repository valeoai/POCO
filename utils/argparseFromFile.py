import sys
from argparse import ArgumentParser, HelpFormatter
import yaml


class ArgumentParserFromFile(ArgumentParser):

    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=[],
                 formatter_class=HelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True):
        super().__init__(prog=prog, usage=usage, description=description, epilog=epilog, parents=parents, formatter_class=formatter_class,
                            prefix_chars=prefix_chars, fromfile_prefix_chars=fromfile_prefix_chars, argument_default=argument_default,
                            conflict_handler=conflict_handler, add_help=add_help, allow_abbrev=allow_abbrev)

        self.file_arg_names=None

    def update_file_arg_names(self, file_arg_names):

        self.file_arg_names=file_arg_names

    def parse(self, use_unknown=False):

        # get the arguments
        if use_unknown:
            args, unknown = self.parse_known_args()
        else:
            args = self.parse_args()
            unknown = None

        config = {}

        # parse the filenames
        if self.file_arg_names is not None:
            for file_arg_name in self.file_arg_names:
                filename = getattr(args, file_arg_name)
                config_ = yaml.load(open(filename, 'r'), Loader=yaml.FullLoader)
                for key, val in config_.items():
                    config[key] = val

        # iterate over the args
        for arg in vars(args):
            config[arg] = getattr(args, arg)

        # use the unknow
        if use_unknown:
            pos = 0
            while(pos < len(unknown)):
                arg = unknown[pos]
                if "--" in arg:
                    key = str(arg[2:])
                    pos += 1
                    try:
                        value = eval(unknown[pos])
                    except:
                        value = unknown[pos]
                    print(key, value)
                    config[key] = value
                pos+=1

        return config