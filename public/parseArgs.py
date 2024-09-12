import argparse
import os


class ParseArgs():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Script-related parameters for VirusImage")
        self.parser.add_argument("--in", type=str, default=None, help="Configuration file of the script")

        self.scripts = ['finetune', 'evaluate', 'image_enhance']

        # general args
        self.parser.add_argument("--datadir", type=str, default=None, help="Root directory of data that needs to be trained(or finetuning) and evaluated.")
        self.parser.add_argument("--datset", type=str, default=None, help="A folder of data that needs to be trained(or finetuning) and evaluated is the name of the data. There should be a corresponding csv file under the folder, named after train, test, val, and evaluate. eg. H1N1 -> train.csv test.csv val.csv evaluate.csv")
        self.parser.add_argument("--gpu", type=str, default='0', help="Index of GPU to use.")
        self.parser.add_argument("--script", type=str, default=None, choices=self.scripts, help="Scripts related to tasks that need to be executed.")

        # enhance script
        self.parser.add_argument("--enhance_list", type=str, default=None, help="The direction that needs to be enhanced. eg. enhance1,enhance2,...")
        self.parser.add_argument("--enhance_save_path", type=str, default=None, help="The location where the enhanced results need to be saved.")
        self.parser.add_argument("--preserve_original_data", type=bool, default=True, help="Whether to retain the original data in the enhanced results.")


        self.argKeys = self.argsToAttr()
        self.parseArgs()

    def argsToAttr(self) -> list:
        args = self.parser.parse_args()
        argKeys = []
        for key, value in vars(args).items():
            setattr(self, key, value)
            argKeys.append(key)
        return argKeys


    def parseArgs(self) -> None:
        if self["in"] is None:
            pass
        else:
            self.parseConfigFile()

    # If --in is set, the corresponding configuration file is parsed.
    def parseConfigFile(self) -> None:
        assert os.path.exists(self["in"]) == True, "{} is not a file.".format(self["in"])
        def getToken(line:str):
            line = line.replace("\n", "").replace("\r", "")
            for i, x in enumerate(line):
                if x != " ":
                    line = line[i:]
                    break
            i = len(line)
            while i > 0:
                if line[i-1] != " ":
                    line = line[:i]
                    break
                i -= 1
                if i == 0:
                    line = ""
            if len(line) == 0:
                return None, None
            
            index = line.find(" ")
            if index <= 0:
                return line, None
            elif index > 0:
                return line[0:index], line[index+1:]
        with open(self["in"], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            token, value = getToken(line=line)
            if token is None and value is None:
                continue
            elif value is None:
                self[token] = True
            else:
                if token == 'enhance_list':
                    value = value.split(",")
                    value = [x for x in value if len(x) > 0]
                self[token] = value



    def __str__(self) -> str:
        argStr = "{\n"
        for key in self.argKeys:
            argStr += "    {} : {},\n".format(key, self[key])
        argStr += "}"
        argStr = argStr.replace(",\n}", "\n}")
        return argStr
    

    def __getitem__(self, key) -> None:
        if key in self.argKeys:
            return getattr(self,key)
        else:
            raise KeyError(f"Key '{key}' not found")
        
    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)
        if key not in self.argKeys:
            self.argKeys.append(key)
        

