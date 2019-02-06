#include <array> 
#include <iostream> 
#include <random> 
#include <vector> 
#include <mpi.h>
#include <Windows.h>
#include <fstream>
#include <string>

const unsigned int field_width = 1 << 10;
const unsigned int field_height = 1 << 10;
const unsigned int generations = 100;
constexpr size_t N_THREADS = 8;

void PrintField(std::ostream& output, const std::vector<char>& field) {
	std::string temp("");
	for (int i = 1; i <= field_height; ++i) {
		for (int j = 1; j <= field_width; ++j) {
			temp += std::to_string((field[i * (field_width + 2) + j] ? 'x' : 'o'));
		}
		temp += "\n";
	}
	output << temp;
}

static_assert(field_height % N_THREADS == 0, "Incorrect field size");

int main(int argc, char* argv[]) {
	std::fstream f;
	constexpr int SEED = 1646868;
	constexpr std::array<size_t, 8> HARDCODE_OFFSET =
	{ 0, 1, 2, field_width + 2, field_width + 4, 2 * field_width + 4, 2 * field_width + 5, 2 * field_width + 6 };

	int numprocs, myid;
	int rc;
	if (rc = MPI_Init(&argc, &argv))
	{
		std::cout << "Îøèáêà çàïóñêà, âûïîëíåíèå îñòàíîâëåíî " << std::endl;
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	std::vector<char> field((field_width + 2) * (field_height + 2));
	std::vector<char> field_history;

	if (myid == 0) {
		std::mt19937 engine(SEED);
		std::uniform_int_distribution<> distribution(0, 1);

		for (int i = 1; i <= field_height; ++i) {
			for (int j = 1; j <= field_width; ++j) {
				field[i * (field_width) + j] = distribution(engine);
			}
		}
		field_history = field;

	}

	for (size_t generation = 0; generation < generations; ++generation) {
		MPI_Bcast(field.data(), field.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
		std::vector<int> sizes(8, (field_width + 2) * field_height / N_THREADS);
		std::vector<char> intermediate((field_width + 2) * field_height / N_THREADS);

		MPI_Reduce_scatter(
			field.data() + HARDCODE_OFFSET[myid],
			intermediate.data(),
			sizes.data(),
			MPI_CHAR,
			MPI_SUM,
			MPI_COMM_WORLD
		);

		for (size_t i = 0; i < intermediate.size(); ++i) {
			if (i % (field_width + 2) == 0 || i % (field_width + 2) == (field_width + 1)) {
				intermediate[i] = 0;
			}
			else {
				if (intermediate[i] == 3 || intermediate[i] == 2 && field[(field_width + 2) * (myid + 1) + i]) {
					intermediate[i] = 1;
				}
				else {
					intermediate[i] = 0;
				}
			}
		}

		MPI_Gather(
			intermediate.data(),
			intermediate.size(),
			MPI_CHAR,
			field.data() + (field_width + 2),
			(field_width + 2) * field_height,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD
		);

		if (myid == 0) {
			// ìåñòî â êîòîðîì íóæíî îñòàíàâëèâàòü èãðó, åñëè äîñòèãíóò êîíåö. Íî íóæíî îñòàíîâèòü âñå ïðîöåññû 
			field_history = field;
			// âìåñòî cout ìîæíî ïåðåäàòü ôàéëîâûé ïîòîê 
			std::string filename(".\\dumps\\gen" + std::to_string(generation) + ".txt");
			f.open(filename.c_str(), std::ios::out);
			//PrintField(std::cout, field);
			PrintField(f, field);
			f.close();
			std::cout << std::endl << std::endl << "GENERATION: " << generation << " ENDED" << std::endl << std::endl;
		}
	}

	MPI_Finalize();
	system("pause");
	return 0;
}
