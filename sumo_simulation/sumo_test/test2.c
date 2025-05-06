

int main(void){
	
	int n, d = 2, resto, conta=0;
	printf("Insrisci un numero intero positivo: ");
	scanf("%d", &n);

	while (d < n && conta==0){
		resto = n % d;
		if (resto == 0)
			conta++;

		d++;
	}

	if(conta == 0)
		printf("Il numero è primo\n");
	else
		printf("Il numero NON è primo\n");

	return 0;
}