// main.cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

// --- Shader Sources ---
// A simple vertex shader that passes through positions and texture coordinates.
const char* vertexShaderSource = R"(
    #version 120
    attribute vec2 position;
    attribute vec2 texCoord;
    varying vec2 vTexCoord;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        vTexCoord = texCoord;
    }
)";

// A fragment shader that applies the HALD CLUT.
// It expects two textures:
//   - "inputImage": the original image
//   - "haldLut": the HALD CLUT image arranged as a (lutSize*lutSize) x (lutSize) image.
// The uniform "lutSize" is the size of one dimension of the cube (computed as cube root of total LUT pixels).
const char* fragmentShaderSource = R"(
    #version 120
    uniform sampler2D inputImage;
    uniform sampler2D haldLut;
    uniform float lutSize;
    varying vec2 vTexCoord;
    void main() {
        // Get the original color
        vec4 color = texture2D(inputImage, vTexCoord);

        // For a HALD image arranged as (lutSize*lutSize) x (lutSize):
        //   total number of pixels = lutSize^3.
        // We compute a 1D index into the LUT from the RGB components.
        // (Assuming the input color is in [0,1])
        float blueScaled = color.b * (lutSize - 1.0);
        // Compute a 1D index that corresponds to the 3D position:
        // index = R_index + G_index * lutSize + B_index * (lutSize * lutSize)
        float index = color.r * (lutSize * lutSize - 1.0)
                    + color.g * ((lutSize - 1.0) * lutSize)
                    + blueScaled;

        // Convert the 1D index into 2D texture coordinates.
        // The haldLut texture has width = lutSize * lutSize and height = lutSize.
        float lutWidth = lutSize * lutSize;
        float x = mod(index, lutWidth);
        float y = floor(index / lutWidth);

        // To sample from the center of the correct texel, add 0.5 and then normalize.
        vec2 lutPos = vec2((x + 0.5) / lutWidth, (y + 0.5) / lutSize);

        // The hardware linear filtering (set up on the texture) will perform interpolation.
        gl_FragColor = texture2D(haldLut, lutPos);
    }
)";

// --- Utility Functions for Shader Compilation ---
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    if (!shader) {
        std::cerr << "Error creating shader!" << std::endl;
        exit(-1);
    }
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if(status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << "Shader compile error: " << buffer << std::endl;
        exit(-1);
    }
    return shader;
}

GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    GLuint program = glCreateProgram();
    if (!program) {
        std::cerr << "Error creating shader program!" << std::endl;
        exit(-1);
    }
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Program link error: " << buffer << std::endl;
        exit(-1);
    }
    // Shaders are no longer needed after linking
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

// --- Utility Function to Create an OpenGL Texture from a cv::Mat ---
// The image is converted from BGR to RGB and flipped vertically to match OpenGL's coordinate system.
GLuint createTexture(const cv::Mat& image) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Use linear filtering so that hardware interpolation handles the trilinear sampling.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Convert BGR (OpenCV default) to RGB.
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    // Flip vertically so that (0,0) is at the bottom left.
    cv::flip(rgb, rgb, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureID;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <LUT_image_path> <photo_path> <output_path>\n";
        return -1;
    }
    std::string lutImagePath = argv[1];
    std::string photoPath    = argv[2];
    std::string outputPath   = argv[3];

    // --- Load Images with OpenCV ---
    cv::Mat lutImage = cv::imread(lutImagePath, cv::IMREAD_COLOR);
    if (lutImage.empty()) {
        std::cerr << "Failed to load LUT image: " << lutImagePath << std::endl;
        return -1;
    }
    cv::Mat inputImage = cv::imread(photoPath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Failed to load input image: " << photoPath << std::endl;
        return -1;
    }

    // Compute the LUT cube size:
    // We assume the LUT image contains exactly (lutSize^3) pixels.
    int totalPixels = lutImage.cols * lutImage.rows;
    int lutSize = std::round(std::cbrt(totalPixels));
    std::cout << "Computed lutSize: " << lutSize << std::endl;

    // --- Initialize GLFW ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }
    // Create a hidden window (offscreen rendering) with dimensions matching the input image.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(inputImage.cols, inputImage.rows, "HALD CLUT", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // --- Initialize GLEW ---
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        glfwTerminate();
        return -1;
    }

    // --- Create Textures ---
    GLuint inputTexture = createTexture(inputImage);
    GLuint lutTexture   = createTexture(lutImage);

    // --- Create Shader Program ---
    GLuint program = createProgram(vertexShaderSource, fragmentShaderSource);
    glUseProgram(program);
    // Set the uniform "lutSize"
    GLint lutSizeLoc = glGetUniformLocation(program, "lutSize");
    glUniform1f(lutSizeLoc, static_cast<float>(lutSize));

    // Bind texture units:
    //   - "inputImage" will be on texture unit 0.
    //   - "haldLut" will be on texture unit 1.
    glUniform1i(glGetUniformLocation(program, "inputImage"), 0);
    glUniform1i(glGetUniformLocation(program, "haldLut"), 1);

    // --- Set Up Geometry (a full-screen quad) ---
    // The quad covers normalized device coordinates [-1,1] and uses texture coordinates [0,1].
    float vertices[] = {
        //  Position      TexCoord
        -1.0f, -1.0f,    0.0f, 0.0f,
         1.0f, -1.0f,    1.0f, 0.0f,
        -1.0f,  1.0f,    0.0f, 1.0f,
         1.0f,  1.0f,    1.0f, 1.0f
    };

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Link vertex attributes.
    GLint posAttrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    GLint texAttrib = glGetAttribLocation(program, "texCoord");
    glEnableVertexAttribArray(texAttrib);
    glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // Set the viewport to match the input image size.
    glViewport(0, 0, inputImage.cols, inputImage.rows);

    // --- Render ---
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glBindVertexArray(vao);
    // Bind the two textures to their respective texture units.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, lutTexture);

    // Draw the quad.
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glfwSwapBuffers(window);

    // --- Read Back the Rendered Image ---
    cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC3);
    glReadPixels(0, 0, inputImage.cols, inputImage.rows, GL_RGB, GL_UNSIGNED_BYTE, outputImage.data);
    // OpenGL’s origin is bottom–left; flip vertically before saving.
    cv::flip(outputImage, outputImage, 0);
    if (!cv::imwrite(outputPath, outputImage))
        std::cerr << "Failed to write output image to " << outputPath << std::endl;
    else
        std::cout << "Output saved to " << outputPath << std::endl;

    // --- Cleanup ---
    glDeleteTextures(1, &inputTexture);
    glDeleteTextures(1, &lutTexture);
    glDeleteProgram(program);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

