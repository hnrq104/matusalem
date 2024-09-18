## Trabalho Final - ICP363 Introdução a Aprendizado de Máquina

**Professor:** João Carlos Pereira da Silva

**Alunos:** 
- Eduardo Monteiro Costa
- Erick Gaiote
- Henrique Cardoso

### Objetivo
Após muito debate e dúvida sobre o que gostaríamos de fazer, decidimos que seria muito interessante estudar modelos generativos de texto - tá na moda.

Para nosso exemplo, iremos tentar construir um
**gerador de versículos da bíblia**.

Inicialmente, estudaremos [RNN's](https://en.wikipedia.org/wiki/Recurrent_neural_network) (LSTM), identificando suas particularidades, características e arquiteturas. Testaremos geração e classificação com elas.

Eventualmente, seria interessante estudar [Transformadores](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)). Tentando compreender o fatídico paper - [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Relatar a evolução em relação as [RNN's](https://en.wikipedia.org/wiki/Recurrent_neural_network) e - se possível com nosso poder computacional - reimplementar o projeto com o uso dessas arquiteturas mais avançadas.

### Roteiro do projeto

#### Parte 1 - Organização e estudo de RNN's
Esta parte está sendo feita agora. A ideia que tivemos é seguir a implementação e a análise feita nesse (antigo - 2015) blog-post https://karpathy.github.io/2015/05/21/rnn-effectiveness/. No entanto, usaremos frameworks diferentes para as redes neurais (pytorch, tensorflow-keras).

Tanto o pytorch, quanto o tensorflow tem implementações do que queremos em seus tutoriais.

- [Tensorflow](https://www.tensorflow.org/text/tutorials/text_generation)
- [Pytorch](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

Durante o treinamento, comparamos a sequência de caractéres prevista pelo modelo com a própria sequência de versículos da Bíblia (ou do texto em questão). Detalhes exatos podem ser vistos no [blog-post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - específicando a estrutura que usaremos backpropagation.

Nesse tipo de modelo, não há etapa de teste, teremos que verificar manualmente as **samples** geradas. Já é possível dizer que não esperamos sentido semântico significativo dos resultados encontrados. Isso acontece pois nosso modelo não será treinado para reconhecer memória semântica das palavras. Usaremos como input uma sequência de caracteres (fixa que definiremos 50,100, etc), e nossa saída será o próximo caractere mais provável segundo o modelo. 

Será interessante ver a evolução do modelo ao treiná-lo com quantidades distintas de épocas (veremos que estruturas sintáticas aparecem).

Inicialmente, os dados de treinamento (um grande arquivo .txt) não precisam de nenhuma formatação. De qualquer forma, escrevi um parser do nosso arquivo da bíblia, para separá-la num csv. Conseguimos então nos perguntar sobre meta-dados de cada versículo do texto. Uma análise boba e incompleta já pode ser vista em `analise_dados.ipnyb`.

***Obs:*** Para a parte de classificação, seguiremos também um [exemplo do Pytorch](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html), e será interessante avaliar os resultados. Pensamos em tentar encontrar o livro baseado no versículo. Como as sequências são significativamente mais longas dos que as que estão no exemplo dado, estamos curiosos para saber o resultado.

### links importantes:
- https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://www.tensorflow.org/text/tutorials/text_generation
- https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
- https://en.wikipedia.org/wiki/Recurrent_neural_network
- https://en.wikipedia.org/wiki/Generative_pre-trained_transformer
- https://en.wikipedia.org/wiki/Gated_recurrent_unit
- https://en.wikipedia.org/wiki/Seq2seq
- https://en.wikipedia.org/wiki/Attention_(machine_learning)
Links dos githubs: [eduardomdc](https://github.com/eduardomdc), [ekegg](https://github.com/EkEgg), [hnrq104](https://github.com/hnrq104).