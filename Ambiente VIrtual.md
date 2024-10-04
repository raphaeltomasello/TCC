####################################
#     CONFIGURAÇÃO POWER SHELL     #
####################################

	Se a política de execução do Power Shell estiver restrita o Python não irá executar no VS Code (que esteja usando o Power Shell como TERMINAL). Para habilitar a execução siga os passos abaixo:

	1- Abra o Power Shell como administrador
	2- Digite o comando: 'Get-ExecutionPolicy' e verifique o estado da política
		(https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy?view=powershell-7.1)
	3- Configure a política para RemoteSigned: 'Set-ExecutionPolicy RemoteSigned'

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
====================================================================================================================================================================

############################
#     AMBIENTE VIRTUAL     #
############################

>> Documentation: https://virtualenv.pypa.io/en/latest/index.html

1- Instalar ambiente virtual: 'pip install virtualenv'
2- Selecionar a pasta do projeto e criar o ambiente virtual dentro, padronizando o nome como 'venv'
3- Comando para criar ambiente virtual: 'virtualenv [nome_da_virtualenv]'; no nosso caso como o padrão será 'venv' -> 'virtualenv venv'
    Este comando criará um ambiente virtual para a versão de Python padrão. Para criar um ambiente com uma versão específica do Python use o comando a seguir:

    'virtualenv -p=[python_installation] [env_name]' ; exemplo: 'virtualenv -p=C:\Python27\python.exe venv'

4- Selecionar ambiente virtual: 
    'source nome_da_virtualenv/bin/activate' (Linux ou macOS)
    'nome_da_virtualenv/Scripts/Activate' (Windows)

    no nosso caso: 'venv/Scripts/Activate'

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
====================================================================================================================================================================