# Contribute to DeepSaki

There are many ways to contribute to DeepSaki and every help is well appreciated! <br>
I especially want to encourage students and those of you eager to learn to code or to advance your skills in the exiting field of deep learning. DeepSaki started as a project to learn and improve my skills and I hope it might help some of you too!

## Make youre first contribution
- Create a feature request
- Report Issues
- Write code
- Add or improve test
- improve the documentation
- contribute example notebooks

## Install the Dev Environment
The eseaist way is to install the provided [docker container](../environment/Dockerfile), which requires you to have docker installed with nvidia gpu support.

1. build the image:
```bash 
$ docker build --build-arg HTTP_PROXY -t deepsaki -f Dockerfile .
```

2. Create an container
```bash
$ docker container create --gpus all --name deepsaki -p 8888:8888 -v ~/git/sascha-kirch/DeepSaki:/deepsaki -it deepsaki
```

3. start container
```bash
$ docker container start -i deepsaki
```


## Before you create a PR
- Review your contribution
- Run unit test with `$> pytest`
- Run the code formater with `$> black .`
- check for errors and fix fixable ones with `$> ruff check --fix .`
