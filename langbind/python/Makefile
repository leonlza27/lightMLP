lmlpcore:
	gcc ../../mlpCCore/matrix/matrix_static.c ../../mlpCCore/mlp/mlp.c ../../mlpCCore/mlp/filedump.c ../../mlpCCore/qfix_ops/qfix_op.c -o liblightmlp.so -fPIC -shared -g

mbppy:
	gcc matrixbp2py.c -fPIC -shared -o libmbp16d.so -L . -llightmlp -g

topycore:
	gcc open2py.c -fPIC -shared -o libcorepy.so -L . -lmbp16d -llightmlp -g
