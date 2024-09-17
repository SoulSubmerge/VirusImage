from public.parseArgs import ParseArgs
from tools.imageEnhance import ImageEnhance
from scripts.finetune import Finetune

def registerScript(args: ParseArgs) -> None:
    dontNoneList = ['datadir','dataset']
    if args.script == "image_enhance":
        dontNoneList = dontNoneList + ['enhance_list','enhance_save_path']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        
        imageEnhance = ImageEnhance(args=args)
        imageEnhance.run()
    elif args.script == "finetune":
        dontNoneList = dontNoneList + ['gpu','image_size', 'only_save_best', 'save_model_dir', 'log_dir', 'model_type', 'loss_function', 'lr', 'weight_decay', 'momentum', 'batch', 'epoch','num_classes']
        for checkArg in dontNoneList:
            assert not (args[checkArg] is None), "{} cannot be None.".format(checkArg)
        print(args)
        finetune = Finetune(args=args)
        finetune.run()




def main():
    args = ParseArgs()
    registerScript(args=args)


if __name__ == '__main__':
    main()



