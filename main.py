from public.parseArgs import ParseArgs







def main():
    args = ParseArgs()
    args.gpu = 1
    args["nGpu"] = 6
    print(args)



if __name__ == '__main__':
    main()
