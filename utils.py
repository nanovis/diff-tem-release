from argparse import ArgumentParser, RawTextHelpFormatter, Action, ArgumentTypeError

from diff_tem.utils.components import Components
import diff_tem


class QueryAction(Action):
    def __call__(self, parser, args, values, option_string=None):
        if not 1 <= len(values) <= 2:
            msg = 'argument --query only support 1 or 2 arguments, which are:\n' \
                  ' <component type>\n' \
                  ' <component type> <parameter>'
            raise ArgumentTypeError(msg)
        setattr(args, self.dest, values)
        component = values[0]
        try:
            component = Components(component)
            print(component.get_description())
            if len(values) == 2:
                print(component.get_parameter_description(values[1]))
        except:
            raise Exception(f"Unknown components: {values[0]}, \n"
                            f"Acceptable components: {[component.value for component in Components]}")
        exit(0)


class DescriptionAction(Action):
    def __init__(self, *args, **kwargs):
        super(DescriptionAction, self).__init__(*args, **kwargs)
        # TODO: Modify this when needed
        self.description = f"TEM-simulator version {diff_tem.VERSION}\n" \
                           f"TEM-simulator is an open source program for simulation" \
                           f"of transmission electron microscope images and tilt series.\n" \
                           f"Usage: TEM-simulator <input file>\n\n" \
                           f"For more information, type\n" \
                           f"TEM-simulator -help\n" \
                           f"or see <http://TEM-simulator.sourceforge.net>.\n\n" \
                           f"Copyright 2008-2010, Hans Rullgard, Stockholm University " \
                           f"and Lars-Goran Ofverstedt, Karolinska Institute, Sweden.\n" \
                           f"TEM-simulator is free software: you can redistribute it " \
                           f"and/or modify it under the terms of the GNU General Public License as published by" \
                           f"the Free Software Foundation, either version 3 of the License, or (at your option) " \
                           f"any later version.\n" \
                           f"TEM-simulator is distributed in the hope that it will be useful, " \
                           f"but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY " \
                           f"or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more " \
                           f"details.\n" \
                           f"You should have received a copy of the GNU General Public " \
                           f"License along with TEM-simulator.  If not, see <http://www.gnu.org/licenses/>.\n"

    def __call__(self, *args, **kwargs):
        print(self.description)
        exit(0)


def get_args():
    parser = ArgumentParser(description="Differentiable TEM simulator",
                            formatter_class=RawTextHelpFormatter)

    component_types_str = "\n".join([component.value for component in Components])
    file_help_msg = f"The input file is a text file which specifies a " \
                    f"number of components to be used in the simulation, and for each component " \
                    f"assigns values to a number of parameters. The input file has the following " \
                    f"structure:\n" \
                    f"=== <component type 1> <component name 1> ===\n" \
                    f"<parameter 1a> = <value 1a>\n<parameter 1b> = <value 1b>\n" \
                    f"=== <component type 2> <component name 2> ===\n" \
                    f"<parameter 2a> = <value 2a>\n<parameter 2b> = <value 2b>\n\n" \
                    f"Lines starting with an = sign, indicate the " \
                    f"beginning of a component, and the lines following it assign values to " \
                    f"parameters in that component.\n" \
                    f"Component names are optional, and have a use " \
                    f"only for some component types. Lines starting with a # sign are treated as " \
                    f"comments and ignored.\n" \
                    f"The following component types can be used:\n" \
                    f"{component_types_str}\n"
    parser.add_argument("input_file", type=str, help=file_help_msg)

    query_help_msg = "Query information about components or parameters\n" \
                     "To query a particular component, type --query <component type>\n" \
                     "To query a particular parameter, type --query <component type> <parameter>\n" \
                     "See the manual for more details.\n"
    parser.add_argument("--query", type=str,
                        action=QueryAction,
                        nargs="+",
                        help=query_help_msg)

    parser.add_argument("-d", "--description", nargs=0, action=DescriptionAction, help="print information about TEM")

    args = parser.parse_args()
    return args
