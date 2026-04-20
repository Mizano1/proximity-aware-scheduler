# =============================================================
#  Makefile - scheduler-sim-final
# -------------------------------------------------------------
#  Targets:
#    make            -> build the simulator (bin/loadbal_sim)
#    make run_fair   -> build + run a small "fair" benchmark
#    make run_large  -> build + run a large-scale benchmark
#    make clean      -> remove build artifacts
# =============================================================

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -I./include

# --- Directory layout ---
SRC_DIR     = src
BIN_DIR     = bin
RESULTS_DIR = results

# --- Build outputs ---
TARGET  = $(BIN_DIR)/loadbal_sim
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)


# --- Default target : build the simulator ---
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build complete. Run ./bin/loadbal_sim"

# Pattern rule : compile each .cpp into a .o
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# --- Convenience run targets ---
run_fair: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) --mode fair --n 1000 --m 200000

run_large: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) --mode large --n 100000 --m 10000000


# --- Cleanup ---
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean run_fair run_large
